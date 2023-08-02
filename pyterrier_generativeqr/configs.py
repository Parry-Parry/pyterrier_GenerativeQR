
creative = {
    'min_length' : 20, 
    'max_new_tokens' : 40, 
    'temperature' : 1.2,
    'do_sample' : True,
    'top_k' : 200,
    'top_p' : 0.92,
    'repetition_penalty' : 2.1,
    'early_stopping' : True
}

deterministic = {
    'min_length' : 20, 
    'max_new_tokens' : 40, 
    'temperature' : 0.3,
    'do_sample' : False,
    'top_k' : 200,
    'top_p' : 0.92,
    'repetition_penalty' : 2.1,
    'early_stopping' : True
}