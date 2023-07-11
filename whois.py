import whois
import itertools
from tqdm import tqdm

chars = 'abcdefghijklmnopqrstuvwxyz0123456789-'
combnation = [''.join(i) for i in itertools.product(chars, repeat=3) if not (i[0] == '-' or i[-1] == '-')]
comb = list(map(lambda x: x + '.com' , combnation))

for c in tqdm(comb):
 try:
  whois(c)
 except Exception as e:
  print(c)