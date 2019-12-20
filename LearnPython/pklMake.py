import pickle

data = {'a': [1, 2.0, 3, 4+6],
        'b': ('String', u'Unicode String'),
        'c': None}
selfref_list = [1,2,3]
selfref_list.append(selfref_list)

output = open('data.pkl', 'wb')

# Pickle dictionary using protocal 0
pickle.dump(data, output)

# Pickle the list using the highest protocol available
pickle.dump(selfref_list, output, -1)

output.close()