import os
import logging

logger = logging.getLogger(os.path.basename(__name__))

def logging_level():
    return logger.getEffectiveLevel()

VALID_CONTEXT_ATTRS = ['memA', 'i0', 'H', 'W', 'L']

class Context:
    pass

def load_location_history(fname):
    H = []
    with open(fname) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            loc, duration = line.split('\t')
            H.append((loc, int(duration)))
    return H

def compute_W(H):
    return [h[1] for h in H]

def prefix_sums(arr):
    ps = [0]
    for x in arr:
        ps.append(ps[-1] + x)
    return ps

def range_sum(ps, start, end):
    # inclusive indices: sum arr[start:end+1]
    return ps[end+1] - ps[start]

def compute_i0(W, rho, ps):
    for i in range(len(W)):
        if range_sum(ps, 0, i) >= rho:
            return i
    return None

def compute_iS(W, rho, i0, ps):
    for i in range(i0 + 1, len(W)):
        if range_sum(ps, i0 + 1, i) >= rho:
            return i
    return None

def compute_iF(W, i0, rho, ps):
    n = len(W)
    for i in range(i0 + 1, n):
        if range_sum(ps, i, n - 1) < rho:
            return i - 1
    return n - 1

def compute_L(H):
    return set(h[0] for h in H)

def initialize_A(H, L, i0, iS):
    memA = {}
    # precompute sums of away times for each l to avoid repeated sums
    for l in L:
        away_time = 0
        for i in range(i0 + 1):
            if H[i][0] != l:
                away_time += H[i][1]
        memA[(i0, l)] = (away_time, [])
    for l in L:
        base_away = memA[(i0, l)][0]
        for i in range(i0 + 1, iS):
            away_time = base_away
            # Add increments from i0+1 to i
            for j in range(i0 + 1, i + 1):
                if H[j][0] != l:
                    away_time += H[j][1]
            memA[(i, l)] = (away_time, [])
    return memA

def compute_Q(i, X):
    Q = []
    ps = X.ps
    for j in range(X.i0, i):
        if range_sum(ps, j + 1, i) >= X.rho:
            Q.append(j)
    if not Q:
        raise Exception(f'No valid Q({i}) could be computed - i_s was incorrect or not checked?')
    return Q

def compute_A(i, l, X, debug=False):
    memA = X.memA
    if (i, l) in memA:
        return memA[(i, l)][0]
    if debug:
        print(f'\tA({i},{l})')
    Q = compute_Q(i, X)
    if debug:
        print(f'\tQ={Q}')
    min_prev_rezes = None
    min_total_cost = float('infinity')
    H = X.H
    for j in Q:
        for loc in X.L:
            prev_rez_cost = compute_A(j, loc, X, debug=debug)
            cur_rez_cost = 0
            for k in range(j + 1, i + 1):
                if debug:
                    print(f'k = {k}')
                if H[k][0] != l:
                    if debug:
                        print(f'{H[k][0]} != {l}')
                    cur_rez_cost += H[k][1]
            total_cost = prev_rez_cost + cur_rez_cost
            if debug:
                print(f'\t({i},{l}) mid ({j},{loc}): total_cost {total_cost} = {prev_rez_cost} + {cur_rez_cost}')
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                min_prev_rezes = [(j, loc)]
            elif total_cost == min_total_cost:
                min_prev_rezes.append((j, loc))
    memA[(i, l)] = (min_total_cost, min_prev_rezes)
    return min_total_cost

def compute_trailing_cost(l, X):
    acc = 0
    H = X.H
    for i in range(X.iF + 1, X.n + 1):
        if H[i][0] != l:
            acc += H[i][1]
    return acc

def reconstruct_residence_history(X, debug=False):
    min_loc = None
    min_away = float('infinity')
    if debug:
        logging.debug('Total away time:')
    for l in X.L:
        dyn_away_time = compute_A(X.iF, l, X, debug=debug)
        trailing_away_time = compute_trailing_cost(l, X)
        away_time = dyn_away_time + trailing_away_time
        if debug:
            print(f'\t{l}\t{away_time}={dyn_away_time}+{trailing_away_time}')
            print_move_history_tree(X.iF, l, X, 2)
        if away_time < min_away:
            min_loc = l
            min_away = away_time
    move_history = [(X.iF, min_loc)]
    while True:
        cur_rez = move_history[0]
        if cur_rez[0] < X.iS:
            break
        else:
            cur_rez = X.memA[cur_rez][1][0]
            move_history.insert(0, cur_rez)
    R = []
    next_move_idx = 0
    while move_history:
        move = move_history.pop(0)
        if R and R[-1][0] == move[1]:
            pass
        else:
            R.append((move[1], next_move_idx))
        next_move_idx = move[0] + 1
    return R

def rezin(H, rho, debug=False):
    L = compute_L(H)
    W = compute_W(H)
    ps = prefix_sums(W)
    i0 = compute_i0(W, rho, ps)
    if i0 is None:
        max_l = None
        max_stay = -1
        for l in L:
            stay_len = sum(h[1] for h in H if h[0] == l)
            if stay_len > max_stay:
                max_l = l
                max_stay = stay_len
        return [(max_l, 0)]
    iS = compute_iS(W, rho, i0, ps)
    if iS is None:
        max_l = None
        max_stay = -1
        for l in L:
            stay_len = sum(h[1] for h in H if h[0] == l)
            if stay_len > max_stay:
                max_l = l
                max_stay = stay_len
        return [(max_l, 0)]
    iF = compute_iF(W, i0, rho, ps)
    X = Context()
    X.memA = initialize_A(H, L, i0, iS)
    X.i0 = i0
    X.iS = iS
    X.iF = iF
    X.H = H
    X.W = W
    X.L = L
    X.rho = rho
    X.n = len(H) - 1
    X.ps = ps
    logger.debug('Context')
    logger.debug(f'\ti0 {i0}')
    logger.debug(f'\tiS {iS}')
    logger.debug(f'\tiF {iF}')
    logger.debug(f'\tH {H}')
    logger.debug(f'\tW {W}')
    logger.debug(f'\tL {L}')
    logger.debug('')
    rez_history = reconstruct_residence_history(X, debug=debug)
    return rez_history

def print_move_history_tree(i, l, X, indent=0):
    cur_rez = (i, l)
    print('\t' * indent, cur_rez)
    if i == X.i0:
        return
    else:
        for last_rez in X.memA[cur_rez][1]:
            print_move_history_tree(*last_rez, X, indent + 1)

def print_interleaved_histories(R, H):
    localH = list(H)
    localR = list(R)
    idx = 0
    while localH:
        if localR and localR[0][1] <= idx:
            print(f' => {localR.pop(0)[0]}')
        else:
            print(f'{localH.pop(0)[0]}\t{localH.pop(0)[1]}') if localH else None
            idx += 1

def main():
    import sys
    logging.basicConfig(level=logging.DEBUG)
    rho = int(sys.argv[1])
    H = load_location_history(sys.argv[2])
    rez_history = rezin(H, rho)
    print('Residence history:', rez_history)
    print('\nInterleaved histories:')
    print_interleaved_histories(rez_history, H)

if __name__ == '__main__':
    main()
