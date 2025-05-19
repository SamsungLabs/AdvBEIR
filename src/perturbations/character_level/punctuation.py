import re
import random
import numpy as np
from collections import deque
from typing import Union

from src.constants import RANDOM_STATE
from src.perturbations.text_perturbation import TextPerturbation

class Punctuation(TextPerturbation):
    """Inserting/deleting/replacing given number (perturbation_strengh) of punctuation marks."""
    TYPE = "characters_punctuation"
    
    def __init__(self, config: dict):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        self.punctuation_marks = [".", ",", "?", "!", ":", ";", "-", "'", '"']
        self.perturbations = ["insert", "delete", "replace"]
        self.modify_weight = config["modify_weight"] 
        self.insert_weight = config["insert_weight"] 
        random.seed(RANDOM_STATE)
    
    def insert(self, query: str, insert_pos: int) -> str:
        query_chars = list(query)
        mark = random.choice(self.punctuation_marks)
        query_chars.insert(insert_pos, mark)
        return "".join(query_chars)

    def replace(self, query: str, replace_pos: int) -> str:
        query_chars = list(query)
        mark = random.choice(self.punctuation_marks)
        query_chars[replace_pos] = mark
        return "".join(query_chars)

    def delete(self, query: str, delete_pos: int) -> str:
        query_chars = list(query)
        query_chars.pop(delete_pos)
        return "".join(query_chars)

    def get_operations_mapping(self, query: str) -> dict[int, set[str]]:
        """Get possible operations that can be performed on each character index of a query"""
        punctuation_positions = [i for i, char in enumerate(query) if char in self.punctuation_marks]
        insert_positions = [i for i, char in enumerate(query) if char.isspace()]
        insert_positions = [0] + insert_positions + [len(query)] # enable inserting at the beginning/end of the query
        operation_map = {}
        
        for pos in insert_positions:
            operation_map[pos] = {"insert"}
        
        for pos in punctuation_positions:
            operation_map[pos] = {"delete", "replace"}
            
        return operation_map 

    def sample_operations(self, operation_map: dict[int, set[str]], num_total) -> deque[dict[str, Union[int, str]]]:
        available_indices = [index for index in operation_map.keys()]
        # increase the probability that the delete/replace will happen (since we have a lot of whitespaces 
        # and insertions dominate the operation_map).
        weights = np.array([self.modify_weight 
                            if operation_map[idx] and operation_map[idx] =={"delete", "replace"} 
                            else self.insert_weight for idx in available_indices], dtype=np.float64)
        probabilities = weights / weights.sum()
        chosen_indices = np.random.choice(available_indices, p=probabilities, size=max(1, num_total), replace=False)
        
        # create a queue with operations
        queue = deque()
        for index in sorted(chosen_indices):
            queue.append({"index": index, "method": random.choice(list(operation_map[index]))})
        return queue

    def apply_operations(self, query: str, sampled_ops: deque[dict[str, Union[int, str]]]) -> str:

        while sampled_ops:
            operation = sampled_ops.popleft()
            operation_index = operation["index"]
            operation_method = operation["method"]
            
            if operation_method == "insert":
                query = self.insert(query, operation_index)
                # update indices in the queue
                for item in sampled_ops:
                    if item["index"] > operation_index:
                        item["index"] += 1
                            
            elif operation_method == "replace":
                query = self.replace(query, operation_index)
            else:
                query = self.delete(query, operation_index)
                # update indices in the queue
                for item in sampled_ops:
                    if item["index"] > operation_index:
                        item["index"] -= 1
        return query
    
    def __call__(self, queries: list[str]) -> list[str]:
        processed_queries = []
        for query in queries:
            # replacing multiple consecutive spaces with only one
            query = str(re.sub(' +', ' ', query))
            query = query.strip()
            operations_map = self.get_operations_mapping(query)
            operations_number = max(int(self.perturbation_strength * len(operations_map)), 1)
            sampled_ops = self.sample_operations(operations_map, operations_number)
            query = self.apply_operations(query, sampled_ops)
            processed_queries.append(query)
        return processed_queries
