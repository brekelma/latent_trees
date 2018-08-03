




mi_mi = ln_convert(mi_bin(samples))
cl_dict = chow_liu(mi_mi)
# dictionary of edges 
h_edges = [ for tup in keys(cl_dict) if (hidden in tup) ]
println("degree of hidden in Chow-Liu : ", length(h_edges))