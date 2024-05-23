from py_eigentrust import EigenTrust

api_key = 'b3NvX2tleV8x'
a = EigenTrust(api_key=api_key)

ijv = [{
  'i': '0x5244736b3a8f898149ae33f013126a20ce7abc62',
  'j': '0xd2362dbb5aa708bc454ce5c3f11050c016764fa6',
  'v': 11.31571
}, {
  'i': '0x976644c7ed9784e5758bb6584dbe3b91420e3463',
  'j': '0x329c54289ff5d6b7b7dae13592c6b1eda1543ed4',
  'v': 269916.08616
}, {
  'i': '0x0154d25120ed20a516fe43991702e7463c5a6f6e',
  'j': '0xf82e119a20c3103ac5dc5b4a0a84776a034a97b5',
  'v': 3173339.366896588
}, {
  'i': '0x9b5ea8c719e29a5bd0959faf79c9e5c8206d0499',
  'j': '0xf82e119a20c3103ac5dc5b4a0a84776a034a97b5',
  'v': 46589750.00759474
}]
scores = a.run_eigentrust(ijv)
print(scores)

pretrust = [{
  'i': '0x5244736b3a8f898149ae33f013126a20ce7abc62',
  'v': 0.1
}, {
  'i': '0x329c54289ff5d6b7b7dae13592c6b1eda1543ed4',
  'v': 0.2
}]
scores = a.run_eigentrust(ijv, pretrust)
print(scores)

scores = a.run_eigentrust_from_csv('example/lt.csv')
print(scores)

scores = a.run_eigentrust_from_csv('example/lt.csv', 'example/pt.csv')
print(scores)

a = EigenTrust(api_key=api_key, alpha=0.01)
scores = a.run_eigentrust_from_csv('example/lt.csv', 'example/pt.csv')
print(scores)