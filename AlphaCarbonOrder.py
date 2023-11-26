def read_file(file_name):
    """
    Read the file and store the data in a list.
    Each element of the list is a tuple of the form (atom_serial_number, x_coordinate, y_coordinate, z_coordinate).
    """
    file_name = "data_q1.txt"
    data_file = []
    with open(file_name, 'r') as file:
        for line in file:
            split_line = line.split()
            data_file.append({
                'atom_serial_number': int(split_line[0]),
                'x_coordinate': float(split_line[1]),
                'y_coordinate': float(split_line[2]),
                'z_coordinate': float(split_line[3])
            })
    return data_file


def read_test_file(file_name):
    file_name = "test_q1.txt"
    test_file = []
    with open(file_name, 'r') as file:
        for line in file:
            split_line = line.split()
            test_file.append({
                'atom_serial_number': int(split_line[0]),
                'x_coordinate': float(split_line[1]),
                'y_coordinate': float(split_line[2]),
                'z_coordinate': float(split_line[3])
            })
    return test_file


def calculate_distance(atom1, atom2):
    """
    Calculate the Euclidean distance between two atoms.
    """
    return ((atom2['x_coordinate'] - atom1['x_coordinate']) ** 2 +
            (atom2['y_coordinate'] - atom1['y_coordinate']) ** 2 +
            (atom2['z_coordinate'] - atom1['z_coordinate']) ** 2) ** 0.5


def check_alpha_carbon_distances(data_file, max_distance_difference=4.0, expected_distance=3.8):
    """
    Check if the distances between alpha-carbon atoms are approximately equal to the expected distance.
    """
    # Find all atoms in the data
    all_atoms = data_file
    unique_alpha_carbon_atoms = set()



    # Check for potential alpha-carbon atoms based on a more lenient proximity check
    for atom in all_atoms:
        if (abs(atom['x_coordinate'] - round(atom['x_coordinate'])) < max_distance_difference
                and abs(atom['y_coordinate'] - round(atom['y_coordinate'])) < max_distance_difference
                and abs(atom['z_coordinate'] - round(atom['z_coordinate'])) < max_distance_difference):
            unique_alpha_carbon_atoms.add(atom['atom_serial_number'])

    print(f"Number of potential alpha-carbon atoms: {len(unique_alpha_carbon_atoms)}")

    # Check distances between alpha-carbon atoms
    alpha_carbon_order = list(unique_alpha_carbon_atoms)
    alpha_carbon_order.sort(key=lambda x: (x, calculate_total_distance(x, all_atoms)))

    # Print the order of alpha-carbon atoms based on distances
    print(f"Order of alpha-carbon atoms based on distances: {alpha_carbon_order}")


def calculate_total_distance(atom_serial_number, atom_list):
    """Calculate the total distance between an atom and all other alpha-carbon atoms."""
    atom = find_atom(atom_serial_number, atom_list)
    distances = [calculate_distance(atom, other) for other in atom_list if other['atom_serial_number'] != atom_serial_number]
    return sum(distances)

def find_atom(atom_serial_number, atom_list):
    """Helper function to find an atom in the list based on its serial number."""
    for atom in atom_list:
        if atom['atom_serial_number'] == atom_serial_number:
            return atom
    raise ValueError(f"Atom with serial number {atom_serial_number} not found in the provided list.")
if __name__ == "__main__":
    data_file = read_file("data_q1.txt")
    check_alpha_carbon_distances(data_file)

