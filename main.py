import pandas as pd
import numpy as np


def north_west_corner(supply, demand):
    # Creates copies of the supply and demand lists to avoid modifying the originals
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    i = 0  # Initial row index for supply
    j = 0  # Initial column index for demand
    bfs = []  # List to store basic feasible solutions (BFS)
    
    # While the BFS list does not contain enough allocations (len(supply) + len(demand) - 1)
    while len(bfs) < len(supply) + len(demand) - 1:
        s = supply_copy[i]  # Supply at the current row
        d = demand_copy[j]  # Demand at the current column
        v = min(s, d)  # Allocate the minimum of supply and demand
        supply_copy[i] -= v  # Decrease the supply by v
        demand_copy[j] -= v  # Decrease the demand by v
        bfs.append(((i, j), v))  # Add the allocation to the BFS
        
        # Move to the next row if the current supply is exhausted
        if supply_copy[i] == 0 and i < len(supply) - 1:
            i += 1
        # Move to the next column if the current demand is exhausted
        elif demand_copy[j] == 0 and j < len(demand) - 1:
            j += 1
    
    return bfs  # Return the list of basic feasible solutions (BFS)

def get_balanced_tp(supply, demand, costs):
    # Calculate the total supply and demand
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    # If supply is less than demand, add a dummy supply with a penalty cost
    if total_supply < total_demand:
        new_supply = supply + [total_demand - total_supply]  # Add dummy supply
        new_costs = costs + [costs*100]  # Add high penalty costs for the dummy supply
        return new_supply, demand, new_costs
    
    # If demand is less than supply, add a dummy demand with zero cost
    if total_supply > total_demand:
        new_demand = demand + [total_supply - total_demand]  # Add dummy demand
        new_costs = costs + [[0 for _ in demand]]  # Add zero costs for the dummy demand
        return supply, new_demand, new_costs
    
    return supply, demand, costs  # If supply and demand are equal, return the original values

def get_us_and_vs(bfs, costs):
    # Initialize the u and v potentials (dual variables)
    us = [None] * len(costs)
    vs = [None] * len(costs[0])
    us[0] = 0  # Set the potential for the first supply to 0
    bfs_copy = bfs.copy()  # Make a copy of the basic feasible solution list
    
    # Loop until all u and v potentials are calculated
    while len(bfs_copy) > 0:
        for index, bv in enumerate(bfs_copy):
            i, j = bv[0]  # Get the position (i, j) of the current basic variable
            if us[i] is None and vs[j] is None:
                continue  # Skip if both u and v are undefined
                
            cost = costs[i][j]  # Get the cost for the current (i, j) pair
            if us[i] is None:
                us[i] = cost - vs[j]  # Calculate u[i] if it is undefined
            else: 
                vs[j] = cost - us[i]  # Calculate v[j] if it is undefined
            
            bfs_copy.pop(index)  # Remove the processed element from the list
            break  # Break out of the loop to process the next basic variable
            
    return us, vs   # Return the calculated u and v potentials

def get_ws(bfs, costs, us, vs):
    # Calculate the reduced costs (w values) for non-basic variables
    ws = []
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            non_basic = all([p[0] != i or p[1] != j for p, _ in bfs])  # Check if (i, j) is not a basic variable
            if non_basic:
                ws.append(((i, j), us[i] + vs[j] - cost))  # Calculate the reduced cost (w[i,j])
    
    return ws  # Return the list of reduced costs

def can_be_improved(ws):
    # Check if any of the reduced costs are positive, indicating that an improvement is possible
    for p, v in ws:
        if v > 0:
            return True
    return False  # No improvements possible if all reduced costs are non-positive

def get_entering_variable_position(ws):
    # Sort the reduced costs by their values and return the position of the most positive reduced cost
    ws_copy = ws.copy()
    ws_copy.sort(key=lambda w: w[1])  # Sort by reduced cost
    return ws_copy[-1][0]  # Return the position of the variable with the highest reduced cost

def get_possible_next_nodes(loop, not_visited):
    # Get the next possible nodes in the loop
    last_node = loop[-1]  # Last node in the current loop
    nodes_in_row = [n for n in not_visited if n[0] == last_node[0]]  # Nodes in the same row
    nodes_in_column = [n for n in not_visited if n[1] == last_node[1]]  # Nodes in the same column
    
    # If the loop has less than 2 nodes, return all possible row and column nodes
    if len(loop) < 2:
        return nodes_in_row + nodes_in_column
    else:
        prev_node = loop[-2]  # Previous node in the loop
        row_move = prev_node[0] == last_node[0]  # Check if the last move was in the same row
        if row_move:
            return nodes_in_column  # If the last move was in the row, continue in the column
        return nodes_in_row  # Otherwise, continue in the row

def get_loop(bv_positions, ev_position):
    # Recursively find a loop starting from the entering variable position
    def inner(loop):
        # If the loop has more than 3 nodes, check if it can be closed
        if len(loop) > 3:
            can_be_closed = len(get_possible_next_nodes(loop, [ev_position])) == 1
            if can_be_closed: return loop  # Return the loop if it can be closed
        
        not_visited = list(set(bv_positions) - set(loop))  # Nodes that have not been visited
        possible_next_nodes = get_possible_next_nodes(loop, not_visited)  # Get possible next nodes
        
        for next_node in possible_next_nodes:
            new_loop = inner(loop + [next_node])  # Recur with the next node in the loop
            if new_loop: return new_loop  # If a valid loop is found, return it
    
    return inner([ev_position])  # Start the loop with the entering variable position

def loop_pivoting(bfs, loop):
    # Perform the pivoting operation to update the BFS after a loop is found
    even_cells = loop[0::2]  # Nodes at even positions in the loop
    odd_cells = loop[1::2]  # Nodes at odd positions in the loop
    get_bv = lambda pos: next(v for p, v in bfs if p == pos)  # Get the value of a basic variable
    
    # Find the leaving variable (with the smallest value in the odd cells)
    leaving_position = sorted(odd_cells, key=get_bv)[0]
    leaving_value = get_bv(leaving_position)
    
    # Create the new BFS after the pivot
    new_bfs = []
    for p, v in [bv for bv in bfs if bv[0] != leaving_position] + [(loop[0], 0)]:
        if p in even_cells:
            v += leaving_value  # Increase the value in even cells
        elif p in odd_cells:
            v -= leaving_value  # Decrease the value in odd cells
        new_bfs.append((p, v))  # Add the updated basic variable to the new BFS
        
    return new_bfs  # Return the new basic feasible solution

def transportation_simplex_method(supply, demand, costs):
    # Balance the supply and demand and adjust the costs
    balanced_supply, balanced_demand, balanced_costs = get_balanced_tp(
        supply, demand, costs
    )
    
    # Main loop to apply the transportation simplex method
    def inner(bfs):
        us, vs = get_us_and_vs(bfs, balanced_costs)  # Calculate u and v potentials
        ws = get_ws(bfs, balanced_costs, us, vs)  # Calculate reduced costs
        if can_be_improved(ws):  # If improvements are possible
            ev_position = get_entering_variable_position(ws)  # Find the entering variable position
            loop = get_loop([p for p, v in bfs], ev_position)  # Find the loop for pivoting
            return inner(loop_pivoting(bfs, loop))  # Perform pivoting and recurse
        return bfs  # If no improvements are possible, return the current BFS
    
    basic_variables = inner(north_west_corner(balanced_supply, balanced_demand))  # Start with the NW corner method
    solution = np.zeros((len(costs), len(costs[0])))  # Initialize the solution matrix
    
    # Fill the solution matrix with the final allocation values
    for (i, j), v in basic_variables:
        solution[i][j] = v

    return solution  # Return the optimal solution

def get_total_cost(costs, solution):
    # Calculate the total transportation cost
    total_cost = 0
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            total_cost += cost * solution[i][j]  # Multiply cost by allocation and sum up
    return total_cost  # Return the total cost


# Load the CSV file using pandas and store it in a DataFrame
df = pd.read_csv('parameters.csv')
# Extract the 'costs' matrix from the DataFrame:
# - All rows except the last one (`df.iloc[:-1]`) because the last row contains the demand.
# - All columns except the first (index column) and last (supply column).
# Convert the selected values into a numpy array of type float.
costs = df.iloc[:-1, 1:-1].to_numpy(dtype=float)

# Extract the 'supply' values from the DataFrame:
# - All rows except the last one (`df.iloc[:-1]`).
# - The last column (`df.iloc[:-1, -1]`) corresponds to supply values.
# Convert these values into a numpy array of type float.
supply = df.iloc[:-1, -1].to_numpy(dtype=float)

# Extract the 'demand' values from the DataFrame:
# - The last row (`df.iloc[-1]`) contains the demand.
# - All columns except the first and last (supply column).
# Convert these values into a numpy array of type float.
demand = df.iloc[-1, 1:-1].to_numpy(dtype=float)

# Check if the sum of supply equals the sum of demand:
if sum(supply) != sum(demand):
    # If the sums are not equal, print an error message.
    print("Error: The sum of supply and demand is not equal.")
else:
    # If the sums are equal, proceed with solving the transportation problem
    solution = transportation_simplex_method(supply, demand, costs)
    
    # Print the solution matrix, which indicates the flow of goods between each supply and demand point.
    print(solution)
    
    # Calculate and print the total transportation cost based on the solution.
    print('Total cost: ', get_total_cost(costs, solution))
