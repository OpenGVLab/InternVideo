from collections import defaultdict

class MemoryPool(object):
    def __init__(self):
        self.cache = defaultdict(dict)

    def update(self, update_info):
        for movie_id, feature_per_movie in update_info.items():
            self.cache[movie_id].update(feature_per_movie)

    def update_list(self, update_info_list):
        for update_info in update_info_list:
            self.update(update_info)

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item)==2:
            return self.cache[item[0]][item[1]]
        return self.cache[item]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key)==2:
            self.cache[key[0]][key[1]] = value
        else:
            self.cache[key] = value

    def __delitem__(self, item):
        if isinstance(item, tuple) and len(item)==2:
            del self.cache[item[0]][item[1]]
        else:
            del self.cache[item]

    def __contains__(self, item):
        if isinstance(item, tuple) and len(item)==2:
            return (item[0] in self.cache and item[1] in self.cache[item[0]])
        return (item in self.cache)

    def items(self):
        return self.cache.items()