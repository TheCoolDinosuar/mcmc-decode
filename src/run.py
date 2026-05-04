plaintext = ""
ciphertext = ""

with open("./data/sample/plaintext.txt") as f:
    plaintext = f.readline()

with open("./data/sample/ciphertext.txt") as f:
    ciphertext = f.readline()

print(plaintext)

from decode import decode

decoded_text = decode(ciphertext, True)
print(decoded_text)
