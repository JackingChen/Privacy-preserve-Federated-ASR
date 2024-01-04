# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:06:04 2023

@author: USER
"""
import functools
import multiprocessing
import os
import time

class Namer:
    """Seperator:("::",":","-","_")"""
    def __init__(self):
        self.function_names = []
        self.funcParameters={}

    def decorator(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.function_names.append(func.__name__)
            return func(*args, **kwargs)
        return wrapper
    def Add_attribute(self,**kwargs):
        current_state=self.function_names[-1]
        for key, val in kwargs.items():
            param_str = f"{key},{val}"
            self.funcParameters[current_state].append(param_str)
    def get_function_names(self):
        return self.function_names
    def get_path(self):
        S_str=[]
        for state in self.function_names:
            if state in self.funcParameters.keys():
                All_pstr="-".join(self.funcParameters[state])
            else:
                All_pstr=state
            S_str.append(All_pstr)
        return S_str
    def pop_function_names(self):
        if self.function_names:
            self.function_names.pop()
    def copy(self):
        new_namer = Namer()
        new_namer.function_names = self.function_names[:]
        return new_namer

global namer
namer = Namer()

@namer.decorator
def _Tordin():
    current_state = namer.get_function_names()
    time.sleep(3)


@namer.decorator
def update_weight():
    save_path=namer.function_names
    print("when step into update_weight",save_path)
    _Tordin()
    save_path=':'.join(namer.function_names)
    save_path=namer.function_names
    print(save_path)
    current_state = namer.get_function_names()

@namer.decorator
def FL_round():
    multiprocessing.set_start_method('spawn', force=True)
    for n in range(1):
        round_n=n
        save_path=namer.function_names
        print("before getting into multiprocess",save_path)
        
        pool = multiprocessing.Pool(processes=2)
        idxs=range(2)
        try:
            final_result = pool.starmap_async(client_train, [(namer,idx) for idx in idxs])
            
        except Exception as e:
            print(f"An error occurred while running local_model.update_weights(): {str(e)}")
        finally:
            final_result.wait()  # wait for all clients end
            results = final_result.get()  # get results
        for idx in range(len(results)):  # for each participated clients
            w = results[idx]
        namer.pop_function_names()  # Remove the last function name ('function_B')
        current_state = namer.get_function_names()  # current state is now ['function_A']
@namer.decorator
def stage1_train():
    centralize()
    FL_round()
    current_state = namer.get_function_names()
    


@namer.decorator
def client_train(namer,idx):
    save_path=namer.function_names
    print("when step into client_train",save_path)
    # global namer
    # namer = namer.copy()  # Create a new instance for each process
    print("process_id:", os.getpid())
    update_weight()
    namer.pop_function_names()  # Remove the last function name ('function_C')

@namer.decorator
def centralize():
    # print(namer.get_function_names())
    current_state = namer.get_function_names()




if __name__ == '__main__':
    # manager = multiprocessing.Manager()
    # namer = manager.Namer()
    stage1_train()

    # print(namer.get_function_names())