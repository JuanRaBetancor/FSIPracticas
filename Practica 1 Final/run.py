# Search methods

import search

ab = search.GPSProblem('O', 'N', search.romania)

print "Busqueda por anchura"
print search.breadth_first_graph_search(ab).path()
print "Busqueda por profundidad"
print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()


print "Ramificacion y acotacion. Debe dar [<Node L>, <Node T>, <Node A>, <Node Z>] y 18 Nodos expandidos"
print search.branch_and_bound(ab).path()


print "Ramificacion y acotacion con subestimacion. Debe dar [<Node L>, <Node T>, <Node A>, <Node Z>] y 6 Nodos expandidos"
print search.branch_and_bound_with_underestimation(ab).path()



#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
