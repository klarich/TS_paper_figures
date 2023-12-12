import random, os


CHUNK_SIZE = 1000
PATH = "m_22bbh_results.csv"

def pick_next_random_line(file, offset):
    file.seek(offset)
    chunk = file.read(CHUNK_SIZE)
    lines = chunk.split(os.linesep)
    # Make some provision in case yiou had not read at least one full line here
    line_offset = offset + len(os.linesep) + chunk.find(os.linesep)
    return line_offset, lines[1]

def get_n_random_lines(path, n):
    lenght = os.stat(path).st_size
    results = []
    result_offsets = set()
    with open(path) as input:
        for x in range(n):
            while True:
                offset, line = pick_next_random_line(input, random.randint(0, lenght - CHUNK_SIZE))
                if not offset in result_offsets:
                    result_offsets.add(offset)
                    results.append(line)
                    break
    return results

if __name__ == "__main__":
    lines = get_n_random_lines(PATH, 10000)
    with open('ROCS_10k_random_from_full_enumeration.csv', 'w') as f:
        for line in lines:
            f.write(f"{line}\n")