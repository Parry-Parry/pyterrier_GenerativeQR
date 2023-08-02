import re
import itertools

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def cut_prompt(output : str, input : str):
    return output[len(input):]

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def concatenate(*lists):
    return itertools.chain.from_iterable(lists)