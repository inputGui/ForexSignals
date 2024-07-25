def find_connected_sinks(file_path):
    # Read the input file and parse the data
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
        objects = [line.strip().split() for line in file]

    # Create a grid and place objects
    grid = {}
    source = None
    sinks = {}
    for obj, x, y in objects:
        x, y = int(x), int(y)
        grid[(x, y)] = obj
        if obj == '*':
            source = (x, y)
        elif obj.isupper():
            sinks[obj] = (x, y)

    # Define pipe connections
    connections = {
        '═': [(1, 0), (-1, 0)],
        '║': [(0, 1), (0, -1)],
        '╔': [(1, 0), (0, 1)],
        '╗': [(-1, 0), (0, 1)],
        '╚': [(1, 0), (0, -1)],
        '╝': [(-1, 0), (0, -1)],
        '╠': [(1, 0), (0, 1), (0, -1)],
        '╣': [(-1, 0), (0, 1), (0, -1)],
        '╦': [(1, 0), (-1, 0), (0, 1)],
        '╩': [(1, 0), (-1, 0), (0, -1)],
        '*': [(1, 0), (-1, 0), (0, 1), (0, -1)],
    }
    for sink in sinks:
        connections[sink] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Perform DFS to find connected sinks
    def dfs(x, y, visited):
        if (x, y) in visited:
            return
        visited.add((x, y))

        current = grid.get((x, y))
        if current not in connections:
            return

        for dx, dy in connections[current]:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) in grid:
                neighbor = grid[(new_x, new_y)]
                if neighbor in connections and (-dx, -dy) in connections[neighbor]:
                    dfs(new_x, new_y, visited)

    # Start DFS from the source
    visited = set()
    dfs(source[0], source[1], visited)

    # Find connected sinks
    connected_sinks = [sink for sink, pos in sinks.items() if pos in visited]

    return ''.join(sorted(connected_sinks))


result = find_connected_sinks('coding_qual_input.txt')
print(result)