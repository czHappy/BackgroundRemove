import  copy
def find_first(minnows):
    list_len = len(minnows)
    for idx in range(list_len):
        if minnows[idx] is not None:
            return idx
    return -1
def sharks_minnows(minnows, sharks):
    idx = find_first(minnows)
    while idx != -1:
        # print(minnows[idx])
        h = copy.deepcopy(minnows[idx])
        minnows[idx] = None #这里会把h也置为空
        h = h - 1

        for i in range(idx+1, len(minnows)):
            if h == 0:
                break
            if minnows[i] is not None and minnows[i] == h:
                h = h-1
                minnows[i] = None

        sharks = sharks - 1
        idx = find_first(minnows)
    return sharks >= 0

minnows = [4,3,1,2,4]
sharks = 2
print(sharks_minnows(minnows, sharks))



def change(amount, denom_list):
    alternatives_list = []
    rec_change(amount, denom_list, [], alternatives_list)
    return alternatives_list



def rec_change(amount, denom_list, change_list, alternatives_list):
    if denom_list:
        denom = denom_list[0]
        extra_change = 1
        for i in denom_list:
            if amount - i == 0:
                alternatives_list.append(change_list + extra_change)
            else:
                rec_change(amount - i, denom_list[1:], change_list + extra_change, alternatives_list)
            #    5
            rec_change(amount, denom_list[1:], change_list, alternatives_list)

from  collections import  defaultdict
def wiz_study_length(prereq_list, final='WIZ90001'):
    semesters = 0; cur_prereqs = [final]
    while cur_prereqs:
        prereqs = defaultdict(list)
        for subject in cur_prereqs:
            prereqs[subject[0]].append(subject[1])
        new_prereqs = []
        for subject in prereq_list:
            if subject in prereqs:
                new_prereqs += prereqs[subject]
        cur_prereqs = new_prereqs; semesters += 1

    return semesters
