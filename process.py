input = open('./data/msr_test_gold.utf8', encoding='utf-8')
output = open('./data/msr_test_gold_char.utf8',mode = 'w', encoding='utf-8')

text = input.read()
bmes_tags = []

for word in text.split():
    if len(word) == 1:
        bmes_tags.append("S")
    else:
        bmes_tags.append("B")
        if len(word) - 2 > 0:
            for i in range(len(word)-2):
                bmes_tags.append("M")
        bmes_tags.append("E")
text = text.replace(" ", "")
text = text.replace("\t", "")
# text = text.replace("\n", "")
print(len(text))
print(len(bmes_tags))
bmes_result = []
index = 0
for i in range(len(text)):
    if text[i] == ' ':
        continue
    if text[i] == '\n':
        bmes_result.append("")
        continue
    bmes_result.append(text[i] + ' ' + bmes_tags[index])
    index+=1
print(index)
res = '\n'.join(bmes_result)
output.write(res)

