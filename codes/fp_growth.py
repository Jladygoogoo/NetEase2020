class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count              # support
        self.parent = parent
        self.next = None               # the same elements
        self.children = {}

    def display(self, ind=1):
        print(''*ind, self.item, '', self.count)
        for child in self.children.values():
            child.display(ind+1)

class FPGrowth:
    def __init__(self, min_support=3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    '''
    Function:  transfer2FrozenDataSet
    Description: transfer data to frozenset type
    Input:  data              dataType: ndarray     description: train_data
    Output: frozen_data       dataType: frozenset   description: train_data in frozenset type
    '''
    def transfer2FrozenDataSet(self, data):
        frozen_data = {}
        for elem in data:
            frozen_data[frozenset(elem)] = 1
        return frozen_data

    '''
      Function:  updataTree
      Description: updata FP tree
      Input:  data              dataType: ndarray     description: ordered frequent items
              FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
              count             dataType: count       description: the number of a record 
    '''
    def updataTree(self, data, FP_tree, header, count):
        # 采用递归，每次插入一个结点
        frequent_item = data[0]
        if frequent_item in FP_tree.children:
            FP_tree.children[frequent_item].count += count
        else:
            FP_tree.children[frequent_item] = FPNode(frequent_item, count, FP_tree)

            # 将头结点列表中的相应item指向该结点，形成链条
            if header[frequent_item][1] is None:
                header[frequent_item][1] = FP_tree.children[frequent_item]
            # 寻找item同级链条的尾部
            else:
                self.updateHeader(header[frequent_item][1], FP_tree.children[frequent_item]) # share the same path

        if len(data) > 1:
            self.updataTree(data[1::], FP_tree.children[frequent_item], header, count)  # recurrently update FP tree

    '''
      Function: updateHeader
      Description: update header, add tail_node to the current last node of frequent_item
      Input:  head_node           dataType: FPNode     description: first node in header
              tail_node           dataType: FPNode     description: node need to be added
    '''
    def updateHeader(self, head_node, tail_node):
        while head_node.next is not None:
            head_node = head_node.next
        head_node.next = tail_node

    '''
      Function:  createFPTree
      Description: create FP tree
      Input:  train_data        dataType: ndarray     description: features
      Output: FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
    '''
    def createFPTree(self, train_data):
        # 初始化头结点列表
        initial_header = {}
        # 遍历数据集，计算item的支持度
        for record in train_data:
            for item in record:
                initial_header[item] = initial_header.get(item, 0) + train_data[record]

        # 利用min_support筛选item，得到新的头结点列表
        header = {}
        for k in initial_header.keys():
            if initial_header[k] >= self.min_support:
                header[k] = initial_header[k]
        # frequent_set表示被保留下来的items集合
        frequent_set = set(header.keys())
        if len(frequent_set) == 0:
            return None, None

        # 更新头结点列表结构，添加指针：{item:(item_support, pointer), ..}
        for k in header:
            header[k] = [header[k], None]

        # 初始化FP-Tree，根结点
        FP_tree = FPNode('root', 1, None)

        # 遍历数据集
        for record, count in train_data.items():
            frequent_item = {}
            # 删除未保留得结点，将保留结点根据支持度重新排序
            for item in record:
                if item in frequent_set:
                    frequent_item[item] = header[item][0]
            if len(frequent_item) > 0:
                ordered_frequent_item = [val[0] for val in sorted(frequent_item.items(), key=lambda val:val[1], reverse=True)]

                # 将更新后的事务插入FP-Tree中
                self.updataTree(ordered_frequent_item, FP_tree, header, count)

        return FP_tree, header

    '''
      Function: ascendTree
      Description: ascend tree from leaf node to root node according to path
      Input:  node           dataType: FPNode     description: leaf node
      Output: prefix_path    dataType: list       description: prefix path
              
    '''
    def ascendTree(self, node):
        prefix_path = []
        while node.parent != None and node.parent.item != 'root':
            node = node.parent
            prefix_path.append(node.item)
        return prefix_path

    '''
    Function: getPrefixPath
    Description: get prefix path
    Input:  base          dataType: FPNode     description: pattern base
            header        dataType: dict       description: header
    Output: prefix_path   dataType: dict       description: prefix_path
    '''
    def getPrefixPath(self, base, header):
        prefix_path = {}
        start_node = header[base][1]
        prefixs = self.ascendTree(start_node)
        if len(prefixs) != 0:
            prefix_path[frozenset(prefixs)] = start_node.count

        while start_node.next is not None:
            start_node = start_node.next
            prefixs = self.ascendTree(start_node)
            if len(prefixs) != 0:
                prefix_path[frozenset(prefixs)] = start_node.count
        return prefix_path

    '''
    Function: findFrequentItem
    Description: find frequent item
    Input:  header               dataType: dict       description: header [name : (count, pointer)]
            prefix               dataType: dict       description: prefix path
            frequent_set         dataType: set        description: frequent set
    '''
    def findFrequentItem(self, header, prefix, frequent_set):
        # for each item in header, then iterate until there is only one element in conditional fptree
        header_items = [val[0] for val in sorted(header.items(), key=lambda val: val[1][0])]
        if len(header_items) == 0:
            return

        for base in header_items:
            new_prefix = prefix.copy()
            new_prefix.add(base)
            support = header[base][0]
            # frequent_set表示
            frequent_set[frozenset(new_prefix)] = support

            prefix_path = self.getPrefixPath(base, header)
            if len(prefix_path) != 0:
                conditonal_tree, conditional_header = self.createFPTree(prefix_path)
                if conditional_header is not None:
                    self.findFrequentItem(conditional_header, new_prefix, frequent_set)

    '''
     Function:  generateRules
     Description: generate association rules
     Input:  frequent_set       dataType: set         description:  current frequent item
             rule               dataType: dict        description:  an item in current frequent item
     '''
    def generateRules(self, frequent_set, rules):
        for frequent_item in frequent_set:
            if len(frequent_item) > 1:
                self.getRules(frequent_item, frequent_item, frequent_set, rules)

    '''
     Function:  removeItem
     Description: remove item
     Input:  current_item       dataType: set         description:  one record of frequent_set
             item               dataType: dict        description:  support_degree 
     '''
    def removeItem(self, current_item, item):
        tempSet = []
        for elem in current_item:
            if elem != item:
                tempSet.append(elem)
        tempFrozenSet = frozenset(tempSet)
        return tempFrozenSet

    '''
     Function:  getRules
     Description: get association rules
     Input:  frequent_set       dataType: set         description:  one record of frequent_set
             rule               dataType: dict        description:  support_degree 
     '''
    def getRules(self, frequent_item, current_item, frequent_set, rules):
        for item in current_item:
            subset = self.removeItem(current_item, item)
            confidence = frequent_set[frequent_item]/frequent_set[subset]
            if confidence >= self.min_confidence:
                flag = False
                for rule in rules:
                    if (rule[0] == subset) and (rule[1] == frequent_item - subset):
                        flag = True

                if flag == False:
                    rules.append((subset, frequent_item - subset, confidence))

                if (len(subset) >= 2):
                    self.getRules(frequent_item, subset, frequent_set, rules)

    '''
      Function:  train
      Description: train the model
      Input:  train_data       dataType: ndarray   description: items
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items   dataType: list      description: frequent items set
    '''
    def train(self, data, display=True):
        data = self.transfer2FrozenDataSet(data)
        FP_tree, header = self.createFPTree(data)
        # FP_tree.display()

        frequent_set = {}
        prefix_path = set([])
        self.findFrequentItem(header, prefix_path, frequent_set)
        rules = []
        # self.generateRules(frequent_set, rules)

        if display:
            print("Frequent Items:")
            frequent_set = sorted(frequent_set.items(), key=lambda x:x[1], reverse=True)
            for item in frequent_set:
                if len(item[0])<=1 or item[1]<=1: continue
                print(item[0], item[1])
            # print("_______________________________________")
            # print("Association Rules:")
            # for rule in rules:
            #     print(list(rule[0])+list(rule[1]),rule[2])
        return frequent_set, rules
