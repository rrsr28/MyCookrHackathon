from datetime import datetime, timedelta
import math
import random

class Order:
    def __init__(self, order_id, kitchen_id, customer_id, pickup_time, location, estimated_prep_time):
        self.order_id = order_id
        self.kitchen_id = kitchen_id
        self.customer_id = customer_id
        self.pickup_time = pickup_time
        self.location = location
        self.estimated_prep_time = estimated_prep_time

def distance_between_locations(location1, location2):
    return math.sqrt((abs(location1[0] - location2[0]))*2 + (abs(location1[1] - location2[1]))*2)

def assign_orders_with_jit(orders, avg_first_mile_time):
    assigned_orders = []
    available_executives = []

    for order in orders:
        # Calculate assignment delay based on estimated preparation time and average first-mile time
        assignment_delay = max(0, order.estimated_prep_time - avg_first_mile_time)

        # Check if there are available executives and they can reach the restaurant in time
        if available_executives and (order.pickup_time - datetime.now() <= timedelta(minutes=assignment_delay)):
            # Assign the order to the executive with the least first-mile time
            nearest_executive = min(available_executives, key=lambda x: distance_between_locations(x.location, order.location))
            assigned_orders.append((order, nearest_executive))
            available_executives.remove(nearest_executive)
        else:
            # Earmark the order and the executive for later assignment
            assigned_orders.append((order, None))
            available_executives.append(order)

    return assigned_orders

def tsp_nearest_neighbor(route):
    # Start with the first order
    current_order = route[0]
    unvisited_orders = route[1:]
    optimized_route = [current_order]

    # Visit each order by selecting the nearest unvisited order until all orders are visited
    while unvisited_orders:
        nearest_order = min(unvisited_orders, key=lambda x: distance_between_locations(current_order.location, x.location))
        optimized_route.append(nearest_order)
        unvisited_orders.remove(nearest_order)
        current_order = nearest_order

    return optimized_route

def optimize_routes(solution):
    optimized_solution = []
    for route in solution:
        optimized_route = tsp_nearest_neighbor(route)
        optimized_solution.append(optimized_route)
    return optimized_solution

def solve_subproblem(orders, max_capacity, max_duration):
    routes = []
    current_route = []
    current_load = 0
    current_time = orders[0].pickup_time  # Start at the first order's pickup time
    
    sorted_orders = sorted(orders, key=lambda x: x.pickup_time)
    
    for order in sorted_orders:
        # Check if adding the order exceeds vehicle capacity or violates time constraints
        if current_route and ((current_load + 1 > max_capacity) or 
                              (current_time + timedelta(minutes=distance_between_locations(current_route[-1].location, order.location)) > 
                               order.pickup_time + timedelta(minutes=max_duration))):
            routes.append(current_route)
            current_route = []
            current_load = 0
            current_time = orders[0].pickup_time  
        
        current_route.append(order)
        current_load += 1
        current_time += timedelta(minutes=distance_between_locations(current_route[-1].location, order.location))
    
    if current_route:
        routes.append(current_route)

    i = 0
    while i < len(routes) - 1:
        current_order = routes[i][-1]
        next_order = routes[i + 1][-1]
        
        if current_order.customer_id == next_order.customer_id and current_order.kitchen_id != next_order.kitchen_id:
            # Merge routes
            routes[i] += routes[i + 1]
            del routes[i + 1]
        elif current_order.customer_id != next_order.customer_id and current_order.kitchen_id != next_order.kitchen_id:
            # Check if the orders are for 2nd customers drop on the way to the 1st customer (vice versa)
            if (routes[i][-1].pickup_time == routes[i + 1][-1].pickup_time) or (
                routes[i][-1].pickup_time <= routes[i + 1][-1].pickup_time + timedelta(minutes=max_duration) and
                routes[i][-1].pickup_time >= routes[i + 1][-1].pickup_time - timedelta(minutes=max_duration)):
                # Merge routes
                routes[i] += routes[i + 1]
                del routes[i + 1]
            else:
                i += 1
        elif current_order.customer_id != next_order.customer_id and current_order.kitchen_id == next_order.kitchen_id:
            # Check if the orders are from the same kitchen and for 2nd customers drop on the way to the 1st customer (vice versa)
            if (routes[i][-1].pickup_time == routes[i + 1][-1].pickup_time) or (
                routes[i][-1].pickup_time <= routes[i + 1][-1].pickup_time + timedelta(minutes=max_duration) and
                routes[i][-1].pickup_time >= routes[i + 1][-1].pickup_time - timedelta(minutes=max_duration)):
                # Merge routes
                routes[i] += routes[i + 1]
                del routes[i + 1]
            else:
                i += 1
        else:
            i += 1
    
    return routes

def decompose_and_solve(orders, max_capacity, max_duration):
    depots = {}
    for order in orders:
        if order.kitchen_id not in depots:
            depots[order.kitchen_id] = []
        depots[order.kitchen_id].append(order)
    
    complete_solution = []
    
    for depot_id, depot_orders in depots.items():
        subproblem_solution = solve_subproblem(depot_orders, max_capacity, max_duration)
        complete_solution.extend(subproblem_solution)
    
    return complete_solution

def batch_orders(orders, max_capacity, max_duration):
    batches = []
    current_batch = []
    current_capacity = 0
    current_time = None
    
    for order in orders:
        # Check if current_batch is not empty before accessing its last element
        if current_batch and current_capacity + 1 <= max_capacity and (not current_time or current_time + timedelta(minutes=distance_between_locations(current_batch[-1].location, order.location)) <= order.pickup_time + timedelta(minutes=max_duration)):
            # Add order to current batch
            current_batch.append(order)
            current_capacity += 1
            current_time += timedelta(minutes=distance_between_locations(current_batch[-1].location, order.location))
        else:
            #a new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [order]
            current_capacity = 1
            current_time = order.pickup_time
            
    if current_batch:
        batches.append(current_batch)
    
    return batches


#function to generate random pickup times
def generate_pickup_time():
    start_time = datetime(2024, 2, 27, 8, 0)  
    end_time = datetime(2024, 2, 27, 20, 0)   
    time_diff = end_time - start_time
    random_time = start_time + timedelta(minutes=random.randint(0, int(time_diff.total_seconds() / 60)))
    return random_time

# Generate a large amount of sample orders
num_orders = 1000
orders = []
for i in range(num_orders):
    order_id = i + 1
    kitchen_id = random.randint(1, 10)  
    customer_id = random.randint(1, 100)  
    pickup_time = generate_pickup_time()
    location = (random.uniform(0, 10), random.uniform(0, 10))  
    estimated_prep_time = random.randint(10, 30)  
    orders.append(Order(order_id, kitchen_id, customer_id, pickup_time, location, estimated_prep_time))


max_capacity = 10  
max_duration = 240  
avg_first_mile_time = 5  


assigned_orders = assign_orders_with_jit(orders, avg_first_mile_time)


assigned_orders_only = [order for order, _ in assigned_orders]


batches = batch_orders(assigned_orders_only, max_capacity, max_duration)

for batch in batches:
    assigned_orders_with_executives = assign_orders_with_jit(batch, avg_first_mile_time)

assigned_orders_only.sort(key=lambda x: x.pickup_time)

assigned_orders_only, executives = zip(*assigned_orders)

solution = decompose_and_solve(assigned_orders_only, max_capacity, max_duration)

optimized_solution = optimize_routes(solution)

#the optimized solution
for i, route in enumerate(optimized_solution):
    print(f"Optimized Route {i+1}:")
    for order in route:
        print(f"Order ID: {order.order_id}, Kitchen ID: {order.kitchen_id}, Customer ID: {order.customer_id}, Pickup Time: {order.pickup_time}, Location: {order.location}")
    print()
