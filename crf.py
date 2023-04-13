import numpy as np
import torch
import sys

mode = 0  # mode 1为train model 0为predict
max_epoch = 20  #最大epoch
loadWeight = 0
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
id2tag = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}

templateFile = "./data/template.utf8"
trainFile = "./data/msr_training_char.utf8"
testFile = "./data/msr_test_gold_char.utf8"
# trainFile = "./data/train1.utf8"
# testFile = "./data/train2.utf8"
modelPath = "./model/past3/CRF-dataSet.model17"
predictPath = "./data/predictSentences3.txt"

class CRF:
    def __init__(self):
        self.learningRate = 0.3
        self.weightMap = {}
        self.UnigramTemplates = []
        self.BigramTemplates = []
        self.loadTemplate()

    # 加载现有模型权重
    def loadWeight(self):
        weightModel = torch.load(modelPath)
        self.weightMap = weightModel['weightMap']
        self.UnigramTemplates = weightModel['UnigramTemplates']
        self.BigramTemplates = weightModel['BigramTemplates']

    # 处理特征模板
    def parseTemplate(self,line):
        tempList = []
        if line.find("/") == -1:
            num = line.split("[")[-1].split(",")[0]
            tempList.append(int(num))
        else:
            left = line.split("/")[0].split("[")[-1].split(",")[0]
            right = line.split("/")[-1].split("[")[-1].split(",")[0]
            tempList.append(int(left))
            tempList.append(int(right))
        return tempList

    # 获取特征模板
    def loadTemplate(self):
        tempFile = open(templateFile, encoding='utf-8')
        # switchFlag True：Bigram ，False：Unigram    先读Unigram，在读Bigram
        switchFlag = False 
        for line in tempFile:
            if line.find("Unigram") > 0 or line.find("Bigram") > 0:
                continue
            if switchFlag:
                tempList = self.parseTemplate(line)
                self.BigramTemplates.append(tempList)
            else:
                if not line.strip():
                    switchFlag = True
                else:
                    tempList = self.parseTemplate(line)
                    self.UnigramTemplates.append(tempList)

    # 获取训练集和测试集
    def getInput(self,path):
        tempFile = open(path, encoding='utf-8')
        sentences = []
        results = []
        sentence = ""
        result = ""
        for line in tempFile:
            line = line.strip()
            if line == "":
                if sentence == "" or result == "":
                    continue
                else:
                    sentences.append(sentence)
                    results.append(result)
                sentence = ""
                result = ""
            else:
                data = line.split(" ")
                sentence += data[0]
                result += data[-1]
        return [sentences, results]

    # 根据特征模板制作特征  
    def makeFeature(self, template, identity, sentence, pos, tag):
        result = ""
        result += identity
        for i in template:
            index = pos + i
            if index < 0 or index >= len(sentence):
                result += " "
            else:
                result += sentence[index]
        result += "/"
        result += tag
        # print(result)
        return result

    # 计算unigram分数
    def calUnigram(self, sentence, pos, curTag):
        score = 0
        template = self.UnigramTemplates
        for i in range(0, len(template)):
            key = self.makeFeature(template[i], str(i), sentence, pos, curTag)
            if key in self.weightMap:
                score += self.weightMap[key]
        return score

    # 计算bigram分数
    def calBigram(self, sentence, pos, preTag, curTag):
        score = 0
        template = self.BigramTemplates
        for i in range(0, len(template)):
            key = self.makeFeature(template[i], str(i), sentence, pos, preTag + curTag)
            if key in self.weightMap:
                score += self.weightMap[key]
        return score
    
    # 修改特征权重
    def calWeight(self, sentence, realRes):
        myRes = self.Viterbi(sentence)
        for i in range(0, len(sentence)):
            tagPredict = myRes[i] 
            tagReal = realRes[i]  
            if tagPredict != tagReal:  
                # Unigram更新
                uniTem = self.UnigramTemplates
                for uniIndex in range(0, len(uniTem)):
                    uniPredict = self.makeFeature(uniTem[uniIndex], str(uniIndex), sentence, i, tagPredict)
                    uniReal = self.makeFeature(uniTem[uniIndex], str(uniIndex), sentence, i, tagReal) 
                    uniGrad = 0
                    if uniPredict not in self.weightMap:
                        self.weightMap[uniPredict] = 0
                        uniGrad += 1
                    else:
                        uniGrad += self.weightMap[uniPredict]
                    if uniReal not in self.weightMap:
                        self.weightMap[uniReal] = 0
                        uniGrad += 1
                    else:
                        uniGrad += self.weightMap[uniReal]
                    # 更新权重
                    self.weightMap[uniPredict] -= uniGrad * self.learningRate
                    self.weightMap[uniReal] += uniGrad * self.learningRate
                # Bigram更新
                biTem = self.BigramTemplates
                for biIndex in range(0, len(biTem)):
                    if i == 0:
                        biPredict = self.makeFeature(biTem[biIndex], str(biIndex), sentence, i, " " + str(tagPredict))
                        biReal = self.makeFeature(biTem[biIndex], str(biIndex), sentence, i, " " + str(tagReal))
                    else:
                        biPredict = self.makeFeature(biTem[biIndex], str(biIndex), sentence, i, myRes[i - 1:i + 1:])
                        biReal = self.makeFeature(biTem[biIndex], str(biIndex), sentence, i, myRes[i - 1:i + 1:])
                    biGrad = 0
                    if biPredict not in self.weightMap:
                        self.weightMap[biPredict] = 0
                        biGrad += 1
                    else:
                        biGrad += self.weightMap[biPredict]
                    if biReal not in self.weightMap:
                        self.weightMap[biReal] = 0
                        biGrad += 1
                    else:
                        biGrad += self.weightMap[biReal]
                    # 更新权重
                    self.weightMap[biPredict] -= biGrad * self.learningRate
                    self.weightMap[biReal] += biGrad * self.learningRate

    # 将BMES格式转换为区间格式
    def tags2Spans(self, tags):
        spans = []
        start = None
        for i, tag in enumerate(tags):
            if tag == "B":
                start = i
            elif tag == "E":
                spans.append([start, i])
                start = None
            elif tag == "S":
                spans.append([i, i])
        return spans

    # 计算相同区间个数
    def countCommonIntervals(self, intervals1, intervals2):
        i = 0
        j = 0
        count = 0
        while i < len(intervals1) and j < len(intervals2):
            if intervals1[i] == intervals2[j]:
                count += 1
                i += 1
                j += 1
            elif intervals1[i][1] < intervals2[j][1]:
                i += 1
            else:
                j += 1
        return count

    # 回溯得到标注序列
    def backPath(self, length, maxScore, preTags):
        resBuf = [""] * length
        scoreBuf = np.zeros(4)
        for i in range(0,4):
            scoreBuf[i] = maxScore[i][length - 1]
        resBuf[length - 1] = id2tag[np.argmax(scoreBuf)]
        for backIndex in range(length - 2, -1, -1):
            resBuf[backIndex] = preTags[tag2id[resBuf[backIndex + 1]]][backIndex + 1]
        res = "".join(resBuf)
        return res
    
    # viterbi算法寻找最优分词法
    def Viterbi(self, sentence):
        length = len(sentence)
        preTags = np.zeros((4, length), dtype=str)
        maxScore = np.zeros((4, length))
        for word in range(0, length):
            for tagid in range(0, 4):
                curTag = id2tag[tagid]
                if word == 0:
                    uniScore = self.calUnigram(sentence, 0, curTag)
                    biScore = self.calBigram(sentence, 0, ' ', curTag)
                    maxScore[tagid][word] = uniScore + biScore
                    preTags[tagid][word] = None
                    # 第一个字母不能为ME
                    maxScore[tag2id['M']][0] = -sys.maxsize - 1
                    maxScore[tag2id['E']][0] = -sys.maxsize - 1
                else:
                    scores = np.zeros(4)
                    for i in range(0, 4):
                        preTag = id2tag[i]
                        preValue = maxScore[i][word - 1]
                        uniScore = self.calUnigram(sentence, word, curTag)
                        biScore = self.calBigram(sentence, word, preTag, curTag)
                        scores[i] = preValue + uniScore + biScore
                        # B后面不能跟BS                       
                        if preTag == 'B' and (curTag == 'B' or curTag == 'S'):
                            scores[i] = -sys.maxsize - 1
                        # M后面不能跟BS
                        if preTag == 'M' and (curTag == 'B' or curTag == 'S'):
                            scores[i] = -sys.maxsize - 1
                        # E后面不能跟ME
                        if preTag == 'E' and (curTag == 'M' or curTag == 'E'):
                            scores[i] = -sys.maxsize - 1
                        # S后面不能跟ME
                        if preTag == 'S' and (curTag == 'M' or curTag == 'E'):
                            scores[i] = -sys.maxsize - 1
                    maxScore[tagid][word] = np.max(scores)
                    preTags[tagid][word] = id2tag[np.argmax(scores)]
        # 最后一个字母不能为BM
        maxScore[tag2id['B']][length-1] = -sys.maxsize - 1
        maxScore[tag2id['M']][length-1] = -sys.maxsize - 1
        res = self.backPath(length, maxScore, preTags)
        return res

    # # viterbi算法寻找最优分词法
    # def Viterbi(self, sentence):
    #     length = len(sentence)
    #     preTags = np.zeros((4, length), dtype=str)
    #     maxScore = np.zeros((4, length))
    #     for word in range(0, length):
    #         for tagid in range(0, 4):
    #             curTag = id2tag[tagid]
    #             if word == 0:
    #                 uniScore = self.calUnigram(sentence, 0, curTag)
    #                 biScore = self.calBigram(sentence, 0, ' ', curTag)
    #                 maxScore[tagid][word] = uniScore + biScore
    #                 preTags[tagid][word] = None
    #             else:
    #                 scores = np.zeros(4)
    #                 for i in range(0, 4):
    #                     preTag = id2tag[i]
    #                     preValue = maxScore[i][word - 1]
    #                     uniScore = self.calUnigram(sentence, word, curTag)
    #                     biScore = self.calBigram(sentence, word, preTag, curTag)
    #                     scores[i] = preValue + uniScore + biScore
    #                 maxScore[tagid][word] = np.max(scores)
    #                 preTags[tagid][word] = id2tag[np.argmax(scores)]
    #     res = self.backPath(length, maxScore, preTags)
    #     return res
    
    #开始训练
    def train(self):
        train_sentences, train_results = self.getInput(trainFile)
        test_sentences, test_results = self.getInput(testFile)
        trainNum = len(train_sentences)
        testNum = len(test_sentences)
        # print(trainNum)
        # print(testNum)
        for epoch in range(0, max_epoch):
            # 训练集
            trainPrecision = 0
            trainRecall = 0
            trainF1 = 0
            for i in range(0, trainNum):
                if i % 10000 == 0:
                    print(f"epoch:{epoch+1}/{max_epoch}" + f" process:{i}/{trainNum}" + (" rate:%.2f%%" % (100 * i/trainNum)))
                sentence = train_sentences[i]
                result = train_results[i]
                self.calWeight(sentence, result)
                myRes = self.Viterbi(sentence)  #训练完修改权重后重新预测
                mySpans = self.tags2Spans(myRes)
                realSpans = self.tags2Spans(result)
                commonSpans = self.countCommonIntervals(mySpans,realSpans)
                P = commonSpans/len(mySpans)
                R = commonSpans/len(realSpans)
                if P ==0 and R == 0:
                    P = 0.1
                    R = 0.1
                if P+R == 0:
                    print(sentence)
                    print(mySpans)
                    print(realSpans)
                    continue
                F = 2*P*R/(P+R)
                trainPrecision += P
                trainRecall += R
                trainF1 += F
            trainPrecision /= trainNum
            trainRecall /= trainNum
            trainF1 /= trainNum
            print("epoch" + str(epoch+1) + ":  P:" + str(float(trainPrecision))
                + ",R:" + str(float(trainRecall)) + ",F1:" + str(float(trainF1)))
            torch.save(
                {
                    'weightMap': self.weightMap,
                    'BigramTemplates': self.BigramTemplates,
                    'UnigramTemplates': self.UnigramTemplates
                },
                "./model/CRF-dataSet.model"+str(epoch+1)
            )
            # 测试集
            testPrecision = 0
            testRecall = 0
            testF1 = 0
            for i in range(0, testNum):
                # if i % 1000 == 0:
                #     print(f"epoch:{epoch+1}/{max_epoch}" + f" process:{i}/{testNum}" + (" rate:%.2f%%" % (100 * i/testNum)))
                sentence = test_sentences[i]
                result = test_results[i]
                myRes = self.Viterbi(sentence)
                mySpans = self.tags2Spans(myRes)
                realSpans = self.tags2Spans(result)
                commonSpans = self.countCommonIntervals(mySpans,realSpans)
                P = commonSpans/len(mySpans)
                R = commonSpans/len(realSpans)
                if P ==0 and R == 0:
                    P = 0.1
                    R = 0.1
                if P+R == 0:
                    print(sentence)
                    print(mySpans)
                    print(realSpans)
                    continue
                F = 2*P*R/(P+R)
                testPrecision += P
                testRecall += R
                testF1 += F
            testPrecision /= testNum
            testRecall /= testNum
            testF1 /= testNum
            print("         " + "P:" + str(float(testPrecision))
                + ",R:" + str(float(testRecall)) + ",F1:" + str(float(testF1)))
            # torch.save(
            #     {
            #         'weightMap': self.weightMap,
            #         'BigramTemplates': self.BigramTemplates,
            #         'UnigramTemplates': self.UnigramTemplates
            #     },
            #     "./model/CRF-dataSet.model"+str(epoch+1)
            # )

    # 预测
    def predict(self, sentence, checkpoint):
        self.weightMap = checkpoint['weightMap']
        self.UnigramTemplates = checkpoint['UnigramTemplates']
        self.BigramTemplates = checkpoint['BigramTemplates']
        # print(self.weightMap)
        # print(self.UnigramTemplates)
        # print(self.BigramTemplates)
        return self.Viterbi(sentence)

# BMES -> 句子
def tag2sentence(sentence,label):
    str = ""
    for i in range(len(label)):
        if label[i] == 'S':
            str += sentence[i]+"/"
        elif label[i] == 'B':
            str += sentence[i]
        elif label[i] == 'M':
            str += sentence[i]
        elif label[i] == 'E':
            str += sentence[i]+"/"
    return str

# 预测输入句子
def predictSentence():
    weightModel = torch.load(modelPath)
    tempFile = open(predictPath, encoding='utf-8')
    for line in tempFile:
        line = line.rstrip('\n')
        label = model.predict(line,weightModel)
        # print(label)
        split_sen = tag2sentence(line,label)
        print(split_sen)

if __name__ == '__main__':
    model = CRF()
    if mode:
        if loadWeight:
            print("load model")
            model.loadWeight()
        model.train()
    else:
        predictSentence()
