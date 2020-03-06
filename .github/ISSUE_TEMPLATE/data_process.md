---
name: Data Process
about: 数据处理工作
title: "[Data]"
labels: 无码
assignees: ''

---

![Untitled Diagram](https://user-images.githubusercontent.com/39988460/75511927-bc26b100-5a2a-11ea-9eec-1581f3c42fcd.jpg)
---
 
进行数据处理时，请按以下顺序进行
- [ ] 数据重命名
  * 命名格式为日期时间编号(时间精确到秒)
- [ ] 更新说明文档
  * 说明文字直接追加到已有的说明文档最后，应包含以下信息
```
    * 数据范围：所标注数据的起始文件名和终止文件名
    * 标注数量：总的标注图片数
    * 标注时间：开始时间和结束时间或者只填结束时间
    * 标注人员：多个人员用英文逗号隔开
    * 标注工具：标注软件名，若用代码合成，则填写函数名
    * 备注：上述信息外的额外内容
```
- [ ] 标注数据
  * 根据任务需求完成相应的标注任务，标注信息以Json格式进行存储
  * 机器自动识别可运行DataProcess.py，但需再次审核数据无误。
- [ ] 上传结果
  * 按照[数据存储规范](https://github.com/evolzed/armlogic/issues/79)将数据上传至GitLab文件夹中
  * 将标注文件上传到Annotation目录中
  * 将重命名的数据上传到Processed目录中
  * 将抠图上传到Foreground或Background目录中

请勾选上述已完成项目,并在该Issues中回复数据范围、标注数量、标注时间、标注人员、标注工具、备注等信息
