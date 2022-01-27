class RsaCrypt:
	def __init__(self, p, q):
		# 2 prime numbers p and q (typically large)
		self.p = p
		self.q = q 
		# compute n = p x q and z = (p - 1) x (q - 1)
		self.n = p * q
		self.z = (p - 1) * (q - 1)
		# choose a number relatively prime to z and call it d
		for i in range(2, self.z):
			if self._gcd(i, self.z) == 1:
				break
		self.d = i
		# find e such that e x d (mod z) = 1 
		for i in range(self.z):
			if ((i * self.d) % self.z) == 1:
				break
		self.e = i

	def encrypt(self, plaintext):
		ciphertext = []
		for p in plaintext:
			ciphertext.append(pow(p, self.e) % self.n)
		return ciphertext

	def decrypt(self, ciphertext):
		plaintext = []
		for c in ciphertext:
			plaintext.append(pow(c, self.d) % self.n)
		return plaintext

	def _gcd(self, num1, num2):
		while (num2 != 0):
			num1, num2 = num2, num1 % num2
		return num1

	def printParams(self):
		print("Printing RSA parameters:")
		print(f"p = {self.p}")
		print(f"q = {self.q}")
		print(f"n = {self.n}")
		print(f"z = {self.z}")
		print(f"d = {self.d}")
		print(f"e = {self.e}")


def main():
	rsa = RsaCrypt(3, 11)
	rsa.printParams()
	plaintext  = [19, 21, 26, 1, 5, 5, 26] # SUZANNE
	ciphertext = rsa.encrypt(plaintext)
	decrypted  = rsa.decrypt(ciphertext)

	print(f"plaintext  = {plaintext}")
	print(f"ciphertext = {ciphertext}")
	print(f"decrypted  = {decrypted}")

if __name__ == '__main__':
	main()
