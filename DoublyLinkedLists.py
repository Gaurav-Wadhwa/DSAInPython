class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def print_list(self):
        temp = self.head
        while temp is not None:
            print(temp.value)
            temp = temp.next
        
    def append(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.length += 1
        return True
    
    def pop(self):
        if self.length == 0:
            return None
        temp = self.tail
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
            temp.prev = None
        self.length -= 1
        return temp
    
    def prepend(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.length += 1
        return True
    
    def pop_first(self):
        if self.length == 0:
            return None
        temp = self.head
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
            temp.next = None
        self.length -= 1
        return temp
    
    def get(self, index):
        if index < 0 or index >= self.length:
            return None
        index_comp = self.length - index - 1
        if index < index_comp:
            temp = self.head
            for _ in range(index):
                temp = temp.next
        else:
            temp = self.tail
            for _ in range(index_comp):
                temp = temp.prev
        return temp
    
    def set_value(self, index, value):
        temp = self.get(index)
        if temp is not None:
            temp.value = value
            return True
        return False
    
    def insert(self, index, value):
        if index < 0 or index > self.length:
            return False
        if index == 0:
            return self.prepend(value)
        if index == self.length:
            return self.append(value)
        new_node = Node(value)
        before = self.get(index - 1)
        after = before.next
        before.next = new_node
        new_node.prev = before
        new_node.next = after
        after.prev = new_node
        self.length += 1
        return True
    
    def remove(self, index):
        if index < 0 or index >= self.length:
            return None
        if index == 0:
            return self.pop_first()
        if index == self.length - 1:
            return self.pop()
        before = self.get(index - 1)
        temp = before.next
        after = before.next.next
        before.next = after
        after.prev = before
        temp.next = None
        temp.prev = None
        self.length -= 1
        return temp

    # Palindrome checker
    def is_palindrome(self):
        mid_index = self.length // 2
        forward = self.head
        backward = self.tail
        for _ in range(mid_index):
            if forward.value != backward.value:
                return False
            forward = forward.next
            backward = backward.prev
        return True

    # Reverse
    def reverse(self):
        temp = None
        current = self.head
        while current is not None:
            current.prev = current.next
            current.next = temp
            temp = current
            current = current.prev
        temp = self.head
        self.head = self.tail
        self.tail = temp
    # Partition List
    def partition_list(self, x):
        if self.length == 0:
            return None
        dummy1 = Node(0)
        dummy2 = Node(0)
        prev1 = dummy1
        prev2 = dummy2
        current = self.head
        while current is not None:
            if current.value < x:
                prev1.next = current
                current.prev = prev1
                prev1 = current
            else:
                prev2.next = current
                current.prev = prev2
                prev2 = current
            current = current.next
        prev1.next = dummy2.next
        prev2.next = None
        if dummy2.next is not None:
            dummy2.next.prev = prev1
        self.head = dummy1.next
        self.head.prev = None

    # Reverse between
    def reverse_between(self, start_index, end_index):
        if start_index < 0 or end_index >= self.length or start_index == end_index:
            return None
        dummy = Node(0)
        dummy.next = self.head
        self.head.prev = dummy
        prev = dummy
        for _ in range(start_index):
            prev = prev.next
        current = prev.next
        for _ in range(end_index - start_index):
            to_move = current.next
            current.next = to_move.next
            if to_move.next is not None:
                to_move.next.prev = current
            to_move.next = prev.next
            prev.next.prev = to_move
            prev.next = to_move
            to_move.prev = prev
        self.head = dummy.next
        self.head.prev = None

    # Swap nodes in pairs
    def swap_pairs(self):
        if self.length == 0:
            return None
        dummy = Node(0)
        dummy.next = self.head
        self.head.prev = dummy
        first = self.head
        while first is not None and first.next is not None:
            temp = first.prev
            second = first.next
            first.next = second.next
            if second.next is not None:
                second.next.prev = first
            second.next = temp.next
            second.prev = temp
            temp.next = second
            first = first.next
        self.head = dummy.next
        self.head.prev = None
