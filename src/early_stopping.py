def eval_stop_condition(stop_condition, cm_reward, i_episode):
    stop = False
    if cm_reward >= stop_condition[0]:
        next_episode = i_episode + 1
        if stop_condition[1] <= next_episode:
            stop = True
    else:
        next_episode = 0
    return stop, next_episode

def on_stop(n):
    print('Early stopping, condition reached after', n, 'episodes')