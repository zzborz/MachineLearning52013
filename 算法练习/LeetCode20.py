"""给定一个只包括'('，')'，'{'，'}'，'['，']'的字符串s ，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
示例
1：
输入：s = "()"
输出：true
示例
2：
输入：s = "()[]{}"
输出：true
示例
3：
输入：s = "(]"
输出：false
提示：
1 <= s.length <= 104
s仅由括号'()[]{}'组成"""
class Solution:
    def isValid(self, s: str) -> bool:
        # 解答区，k神解答

        # 创建一个字典dic，用于字符串中【右括号与左括号是否匹配】
        # 里面还有多加了一个【'?': '?'】的键值对，这个也是用于【字符串第一个元素是右括号】的情况，拿来配合下面【栈】中的初始元素【'?'】
        # （因为【dic[stack.pop()]】会拿出【'?'】去匹配字典dic里面的值，所以需要自己设计一个这样的键值去配合这种情况的匹配，其实只要弄一个不是【'}'、']'、')'】的值就好了）
        dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        # 创建一个列表，用作【栈】。里面先加入一个字符【？】，用于防止【字符串第一个元素是右括号】时，使用【stack.pop()】会报错的问题
        stack = ['?']
        # 遍历这个字符串
        for c in s:
        # 如果字符串中的这个元素在字典中，那就把这个元素添加到stack栈中
        #因为字典除【'?'】外，键都是左括号，所以这里的意思其实是说，如果这个c字符是【左括号】，那么就把它添加到stack中
            if c in dic:stack.append(c)
        # 如果字符串中的这个元素不在字典中，那就把stack栈中的最后一个元素 在字典中匹配的值 与c这个字符做匹配，不相等的话，说明这两个字符是不匹配的，也就是不符合要求，所以返回False
        #   因为stack中除了【'?'】外，上方添加的元素都是【左括号】，所以进入个判断的都是【右括号】。把最后一个左括号在字典中对应的值，也就是【该左括号相对应的右括号】，与该【右括号】相对比，不相等的话，
        #   那就有括号不满足题目要求了，所以提前返回False
        #   在字符串第一个字符为【右括号】时，也就是【c not in dic】，会进入该判断。这时的【stack.pop()】，是我们先前填充的【'?'】，【dic['?']】在字典中匹配的值，绝对不等于该【右括号】，
        #   所以直接返回False。其实就是【第一个字符为右括号】时，这个字符串一定不满足题目要求
            elif dic[stack.pop()] != c: return False
        # 整个字符串遍历后，会出现两种情况
        #   1.字符串是符合题目要求的，也就是【stack中所有括号都匹配成功了】，那最后就只剩下了原先填充的【'?'】，这时应该要返回的是True，这里len(stack) == 1结果就是返回True
        #   2.字符串是不符合题目要求的，这种会跳出for循环的情况是【最后一个字符为左括号】，这时除了这个【左括号】外，其他字符都匹配完成了，也就是stack中只有【'?','左括号'】这两个元素了，
        #   这时应该要返回的是False，这里len(stack) == 1的结果就是False
        return len(stack) == 1