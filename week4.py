file = input("Enter the filename you want to read: ")

try:
    with open(file, "r") as infile:
        content = infile.read()

    modified_content = content.upper()

    with open("modified_" + file, "w") as outfile:
        outfile.write(modified_content)

    print(f"Modified content written to modified_{file}")

except FileNotFoundError:
    print("❌ Error: The file does not exist.")
except PermissionError:
    print("❌ Error: You do not have permission to read this file.")
