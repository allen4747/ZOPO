import numpy as np
import torch
from learner_diag import GP_NTK, Network
import numpy as np

def select_query(queries, greedy=False):
    xs = [q[0] for q in queries]
    ys = [np.asarray(q[1]).item() for q in queries]

    if greedy:
        ind = np.argmax(ys)
    else:
        ind = -1
    return xs[ind], ys[ind]

# =========================
# SAM Optimizer Wrapper
# =========================
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) wrapper.
    base_optimizer_cls: torch.optim.Adam 같은 optimizer 클래스를 넣어줌.
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        # base optimizer 생성
        base_optimizer = base_optimizer_cls(params, **kwargs)

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.state = self.base_optimizer.state

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
    @torch.no_grad()
    def _grad_norm(self):
        # 모든 파라미터 gradient의 L2 norm 계산
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.adaptive:
                    e_w = torch.abs(p) * p.grad
                else:
                    e_w = p.grad
                norms.append(torch.norm(e_w, p=2))
        if len(norms) == 0:
            return torch.tensor(0., device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        SAM 1번째 스텝: epsilon 방향으로 weight를 살짝 이동 (w ← w + epsilon)
        """
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            return
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.adaptive:
                    e_w = torch.abs(p) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        SAM 2번째 스텝: w를 원래 위치로 되돌린 뒤(base_optimizer로) 실제 업데이트
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, *args, **kwargs):
        raise NotImplementedError(
            "SAM: step() 대신 first_step(), second_step()를 사용하세요."
        )

    def zero_grad(self):
        self.base_optimizer.zero_grad()

# =========================
# ZOPO Algorithm
# =========================
class ZOPO:
    """Zeroth-Order Prompt Optimization using GP-NTK"""

    def __init__(self, zopo_opts):
        self.lr = zopo_opts['lr']
        self.tolerance = zopo_opts['tolerance']
        self.gd_iters = zopo_opts['maxiter']
        self.max_steps = zopo_opts['max_steps']
        self.neighbors = zopo_opts['neighbors']
        self.nn_depth = zopo_opts['nn_depth']
        self.nn_width = zopo_opts['nn_width']
        self.gp_queries = zopo_opts['gp_queries']
        self.uncertainty_count = zopo_opts['uncertainty_count']
        self.uncertainty_thred = zopo_opts['uncertainty_thred']
        self.input_dim = zopo_opts['input_dim']

        self.sam_rho = zopo_opts.get('sam_rho', 0.05) # SAM rho 값, 기본값 0.05

        self.non_imp_iters = 0
        self.max_fx = None
        self.k = 0
        self.emb_queries = []
        self.api = None
        self.start = True

        self.count = 0

    def normalize(self, x):
        x = (x - self.lb) / (self.ub - self.lb)
        return x

    def init_query(self, init_emb_queries):
        """initialization"""
        self.emb_queries = init_emb_queries
        self.k += len(init_emb_queries)
        next_emb, value = select_query(self.emb_queries, greedy=True)
        gp_input = torch.from_numpy(next_emb.copy()).cuda().float()

        # Adam <-> SAM
        base_opt = torch.optim.Adam
        optimizer = SAM([gp_input], base_optimizer_cls=base_opt, rho=self.sam_rho, lr=self.lr)
        self.opt_state = optimizer.state_dict()
        self.max_fx = value

        func = Network(self.input_dim, hidden_size=self.nn_width, depth=self.nn_depth)
        func.cuda()
        self.gp_ntk = GP_NTK(network = func)
    
    def fit_gp(self, target_x):
        # TODO: normalize
        xs = np.array([q[0].tolist() for q in self.emb_queries])
        ys = np.array([np.asarray(q[1]).item() for q in self.emb_queries])
        dists = np.linalg.norm(xs - target_x, axis=1)
        if len(dists) <= self.gp_queries:
            idx = np.arange(len(dists))
        else:
            idx = np.argpartition(dists, self.gp_queries)[:self.gp_queries]
        xs, ys = xs[idx], ys[idx]


        inputs = torch.from_numpy(xs).float().cuda()
        self.ub = torch.max(inputs, dim=0)[0]
        self.lb = torch.min(inputs, dim=0)[0]
        conf_idx = self.ub == self.lb
        self.ub[conf_idx] = 1
        self.lb[conf_idx] = 0
        inputs = self.normalize(inputs)
        outputs = torch.from_numpy(ys).float().cuda()
        self.gp_ntk.fit(inputs, outputs)

    def discrete_gd(self, updated_emb, instruct_emb_pairs):
        closest_instruct = None
        closest_emb = None
        min_dist = float('inf')

        start_emb = updated_emb.clone().detach()
        for next_emb, next_instruct in instruct_emb_pairs.items():
            next_emb = torch.tensor(next_emb).cuda().half()
            dist = torch.norm(next_emb - start_emb)
            if dist < min_dist and (next_instruct[0] not in self.api.prompts_set.keys()):
                min_dist = dist
                closest_instruct = next_instruct
                closest_emb = next_emb

        return closest_instruct, closest_emb
    
    def get_local_points(self, instruct_emb_pairs, target_x, N_NEIGHBORS):
        xs = np.array([q for q in instruct_emb_pairs.keys()])
        dists = np.linalg.norm(xs - target_x, axis=1)
        idx = np.argpartition(dists, N_NEIGHBORS)[:N_NEIGHBORS]

        for emb in xs[idx]:
            instruct = instruct_emb_pairs[tuple(emb.tolist())]
            if instruct[0] not in self.api.prompts_set.keys():
                print('-------- Local Exploration --------')
                dev_score = self.api.eval_instruct(instruct)
                self.emb_queries += [(emb, dev_score[0])]
                self.k += 1

    def ask(self, instruct_emb_pairs):
        """generate next query"""
        self.k += 1
        steps = 0
        next_emb, value = select_query(self.emb_queries, greedy=self.start)
        instruction = instruct_emb_pairs[tuple(next_emb.tolist())]
        closest_emb = torch.from_numpy(next_emb.copy()).cuda().float()

        # re-init optimizer because of the inaccurate derivative estimation
        if self.max_fx < value:
            self.max_fx = value
            self.non_imp_iters = 0
        else:
            self.non_imp_iters += 1
        if self.non_imp_iters > self.tolerance:
            next_emb, value = select_query(self.emb_queries, greedy=True)
            self.non_imp_iters = 0
            gp_input = torch.from_numpy(next_emb.copy()).cuda().float()
            base_opt = torch.optim.Adam
            optimizer = SAM([gp_input], base_optimizer_cls=base_opt, rho=self.sam_rho, lr=self.lr)
            self.opt_state = optimizer.state_dict()
        try:
            self.fit_gp(next_emb)
        except:
            print('Singular Values!')

        # Adam <-> SAM
        gp_input = torch.from_numpy(next_emb.copy()).cuda().float()
        base_opt = torch.optim.Adam
        optimizer = SAM([gp_input], base_optimizer_cls=base_opt, rho=self.sam_rho, lr=self.lr)

        while True:
            # ===== 1st gradient (현재 위치) =====
            gp_input.requires_grad_(True)
            # self.api.model.zero_grad()
            if gp_input.grad is not None:
                gp_input.grad.zero_()
            self.gp_ntk.net.zero_grad()

            final_val = -self.gp_ntk.pred(self.normalize(gp_input.view(1,-1)))

            (grad1,) = torch.autograd.grad(final_val, gp_input, retain_graph=False)
            gp_input.grad = grad1.detach()
            gradient = gp_input.grad.detach().clone()

            grads_uncertainty = abs(self.gp_ntk.pred_var(self.normalize(gp_input.view(1,-1))))
            print('uncertainty: ',  grads_uncertainty.item())

            if (grads_uncertainty > self.uncertainty_thred) and (steps == 0):
                self.count += 1
            if (grads_uncertainty < self.uncertainty_thred) and (steps == 0):
                self.count = 0

            #  gradient 0이거나, uncertainty 기준으로 local exploration으로 전환
            if (gradient[0].sum() == 0) or ((self.count >= self.uncertainty_count) \
                and (self.k <= self.gd_iters - self.neighbors)):

                self.non_imp_iters = 0
                self.count = 0
                self.get_local_points(instruct_emb_pairs, next_emb, self.neighbors)
                self.fit_gp(next_emb)
                continue

            update = (grads_uncertainty < self.uncertainty_thred) or (steps == 0)
            if update and steps < self.max_steps:
                # ===== SAM 1st step: perturbation =====
                optimizer.load_state_dict(self.opt_state)
                optimizer.first_step(zero_grad=True)

                # ===== 2nd gradient: perturbed 위치에서 다시 gradient =====
                gp_input.requires_grad_(True)
                if gp_input.grad is not None:
                    gp_input.grad.zero_()
                self.gp_ntk.net.zero_grad()

                final_val_pert = -self.gp_ntk.pred(self.normalize(gp_input.view(1,-1)))
                (grad2,) = torch.autograd.grad(final_val_pert, gp_input, retain_graph=False)
                gp_input.grad = grad2.detach()

                # ===== SAM 2nd step: 원위치 복원 + Adam 업데이트 =====
                optimizer.second_step(zero_grad=True)
                self.opt_state = optimizer.state_dict()

                # SAM 업데이트된 연속 embedding 기준으로 discrete GD
                closest_instruct, closest_emb_ = self.discrete_gd(
                gp_input, instruct_emb_pairs)
                if closest_emb_ is None:
                    break
                else:
                    closest_emb = closest_emb_
                instruction = closest_instruct
                steps += 1

                # optimizer가 들고 있는 gp_input 파라미터는 그대로 두고
                # data만 discret embedding으로 덮어씌우기
                with torch.no_grad():
                    gp_input.data.copy_(closest_emb.detach().clone().cuda().float())

                if (grads_uncertainty >= self.uncertainty_thred):
                    break
            else:
                break

        return instruction, closest_emb

    def tell(self, emb, value):
        """update the query and value"""
        self.start = False
        self.emb_queries+= [(emb.detach().cpu().numpy(), value)]

    def stop(self):
        """whether the query budget is met"""
        return self.k >= self.gd_iters
