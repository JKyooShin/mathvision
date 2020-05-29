import random

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    data = list(data)
    F=[]
    for i in range(max_iterations):
        t=[]
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                t.append(data[j])
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            F = t
            if ic > goal_inliers and stop_at_goal:
                break

    return best_model, best_ic, F
