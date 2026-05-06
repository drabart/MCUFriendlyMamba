import argparse

parser = argparse.ArgumentParser(description="Generate include/data.h with const and optional mutable data arrays")
parser.add_argument("--size", "-s", type=int, default=50000, help="Data size (default: 50000)")
parser.add_argument("--mutable", "-m", action="store_true", help="Also generate mutable (data_mut) array")
args = parser.parse_args()

DATA_SIZE = args.size

with open("main/include/data.h", "w+") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write("#define DATA_SIZE " + str(DATA_SIZE) + "\n\n")

    # Always generate const data_const
    f.write("const uint8_t data_const[DATA_SIZE] = {\n")
    for i in range(DATA_SIZE):
        f.write("    " + str(i % 256) + ",\n")
    f.write("};\n\n")

    f.write("extern const uint8_t data_const[DATA_SIZE];\n\n")

    # Optionally generate mutable data_mut
    if args.mutable:
        f.write("#define DATA_MUT_AVAILABLE\n\n")
        f.write("uint8_t data_mut[DATA_SIZE] = {\n")
        for i in range(DATA_SIZE):
            f.write("    " + str(i % 256) + ",\n")
        f.write("};\n\n")

        f.write("extern uint8_t data_mut[DATA_SIZE];\n\n")

arrays_str = "data_const" + (" + data_mut" if args.mutable else "")
print(f"Wrote main/include/data.h with {arrays_str} (DATA_SIZE={DATA_SIZE})")