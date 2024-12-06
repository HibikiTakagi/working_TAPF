import random

class TaskManager():
    def __init__(self, len_node, agents_num, node_distances, assign_policy):
        self.len_node = len_node
        self.agents_num = agents_num
        self.assigned_tasks = set()
        self.assigned_tasks_num = agents_num
        self.not_assigned_tasks = set()
        self.not_tasks_node = set(range(len_node))
        self.reserved_tasks = set()
        self.assigned_task_checker = [-1] * agents_num
        self.reserved_task_checker = [-1] * agents_num
        self.node_distances = node_distances
        self.assign_policy = assign_policy
        self.completetasks = 0

    def reset_tasks(self):
        '''
        self.assigned_tasks.clear()
        self.not_assigned_tasks.clear()
        self.not_tasks_node = set(range(len_node))
        self.reserved_tasks.clear()
        self.completetasks = 0
        '''
        selected_tasks = random.sample(list(self.not_tasks_node), self.agents_num*3)
        self.not_tasks_node = self.not_tasks_node - set(selected_tasks)
        self.not_assigned_tasks |= set(selected_tasks)

    def update_tasks(self, robot_id):
        complete_task = self.assigned_task_checker[robot_id]
        if complete_task == -1:
            '''
            if robot_id == 3:
                print(f"エージェント{robot_id}は完了するタスクを持っていません：")
            '''
            return    
        '''
        if robot_id == 3:
            print(f"エージェント{robot_id}が次のタスクを完了しました：", complete_task)
        '''
        self.assigned_tasks.remove(complete_task)
        self.assigned_task_checker[robot_id] = -1
        new_task = random.choice(list(self.not_tasks_node))
        self.not_tasks_node.remove(new_task)
        self.not_assigned_tasks.add(new_task)
        self.not_tasks_node.add(complete_task)
        self.completetasks += 1
    
    def reserve_next_task(self, robot_id, state_node = None):
        if self.assign_policy == "random_mode":
            return self.reserve_next_task_random(robot_id)
        elif self.assign_policy == "classic_mode":
            return self.reserve_next_task_classic(robot_id, state_node)
        else:
            raise ValueError(f"Unknown assign policy: {self.assign_policy}")
    
    def reserve_next_task_random(self, robot_id):
        if self.reserved_task_checker[robot_id] != -1:
            '''
            if robot_id == 3:
                print(f"エージェント{robot_id}はノード{self.reserved_task_checker[robot_id]}を予約しています。")
            '''
            return self.reserved_task_checker[robot_id]
        reserve_task  = random.choice(list(self.not_assigned_tasks))
        self.not_assigned_tasks.remove(reserve_task)
        self.reserved_tasks.add(reserve_task)
        self.reserved_task_checker[robot_id] = reserve_task
        '''
        if robot_id == 3:
            print(f"エージェント{robot_id}がノード{reserve_task}を予約しました。")
        '''
        return reserve_task

    def reserve_next_task_classic(self, robot_id, state_node):
        if self.reserved_task_checker[robot_id] != -1:
            return self.reserved_task_checker[robot_id]
        #reserve_task  = random.choice(list(self.not_assigned_tasks))
        robot_task_dist_min = self.len_node #infの代用
        reserve_task = self.len_node #nilの代わり
        for reserve_task_cand in self.not_assigned_tasks:
            robot_task_dist = self.node_distances[state_node][reserve_task_cand]
            if robot_task_dist < robot_task_dist_min:
                robot_task_dist_min = robot_task_dist
                reserve_task = reserve_task_cand
                
        self.not_assigned_tasks.remove(reserve_task)
        self.reserved_tasks.add(reserve_task)
        self.reserved_task_checker[robot_id] = reserve_task
        return reserve_task

    def reserved_task_assign(self, robot_id):
        assigned_task = self.assigned_task_checker[robot_id]
        reserved_task = self.reserved_task_checker[robot_id]
        if assigned_task != -1:
            '''
            if robot_id == 3:
                print(f"エージェント{robot_id}は先に{assigned_task}を完了してください。")
            '''
            return
        if reserved_task == -1:
            '''
            if robot_id == 3:
                print(f"エージェント{robot_id}は予約を先にしてください。")
            '''
            return
        self.reserved_tasks.remove(reserved_task)
        self.reserved_task_checker[robot_id] = -1
        self.assigned_tasks.add(reserved_task)
        self.assigned_task_checker[robot_id] = reserved_task
        '''
        if robot_id == 3:
            print(f"エージェント{robot_id}に予約済みノード{reserved_task}を割当てました。")
        '''
        return reserved_task

    def cancel_reserved_task(self, robot_id):
        reserved_task = self.reserved_task_checker[robot_id]
        if reserved_task == -1:
            print("キャンセルできる予約がありません。")
            return
        self.reserved_tasks.remove(reserved_task)
        self.reserved_task_checker[robot_id] = -1
        self.not_assigned_tasks.add(reserved_task)

    '''
    def delete_reserved_task(self, robot_id):
        for reserved_task in self.reserved_tasks:
            if reserved_task[0] == robot_id:
                self.reserved_tasks.remove(reserved_task)
                self.not_assigned_tasks.add(reserved_task[1])
                return reserved_task[1]
        print("ReserveTaskNilError!!")

    def virtual_task_assign(self):
        return random.choice(list(self.not_assigned_tasks))
    
    def task_random_assign(self, robot_id):
        for reserved_task in self.reserved_tasks:
            if reserved_task[0] == robot_id:
                self.reserved_tasks.remove(reserved_task)
                self.assigned_tasks.add(reserved_task[1])
                return reserved_task[1]
        new_assign_task = random.choice(list(self.not_assigned_tasks))
        self.not_assigned_tasks.remove(new_assign_task)
        self.assigned_tasks.add(new_assign_task)
        return new_assign_task

    def task_classic_assign(self, robot_id, robot_env):
        return
        robot_envs[agent_id].curr_state[0](=state_node?)
        を用いて、マンハッタン距離が最も近いタスクを割り当てる。
        距離を計算するメソッドはどこかにあったはず。
        CLASSIC.pyやCONST.pyのshortest_path_matrix等が参考になるかも。

    def shuffle_node(self, len_node):
        tasks = list(range(0, len_node))
        random.shuffle(tasks)
        tasks_2d = [[task] for task in tasks]
        return tasks_2d
    '''

if __name__ == "__main__":
    task_manager = TaskManager(12, 5)
    task_manager.reset_tasks()
    print("―――初期状態―――")
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)

    print("―――タスク予約(その1)―――")
    print("予約するタスク:", task_manager.reserve_next_task(1))
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)

    print("―――タスク予約(その2)―――")
    print("予約するタスク:", task_manager.reserve_next_task(4))
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)

    print("―――タスク割り当て(その1)―――")
    print("新しい割り当てタスク：", task_manager.reserved_task_assign(0))
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)

    print("―――タスク割り当て(その2)―――")
    print("新しい割り当てタスク：", task_manager.reserved_task_assign(1))
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)

    print("―――タスク完了(その1)―――")
    robot_id = 0
    #complete_task = random.choice(list(task_manager.assigned_tasks))
    task_manager.update_tasks(robot_id)
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)
    
    print("―――タスク完了(その2)―――")
    robot_id = 1
    task_manager.update_tasks(robot_id)
    print("ノーマルのノード：", task_manager.not_tasks_node)
    print("割当てがまだのタスクノード：", task_manager.not_assigned_tasks)
    print("割当てが予約されているタスクノード：", task_manager.reserved_tasks)
    print("割当てが予約されているタスクノード(辞書):", task_manager.reserved_task_checker)
    print("割当て済みのタスクノード：", task_manager.assigned_tasks)
    print("割当て済みのタスクノード(辞書):", task_manager.assigned_task_checker)

    print("―――最終結果―――")
    print("完了タスク数：", task_manager.completetasks)