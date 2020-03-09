#本文件示例excel API的用法
import xlsxwriter
excelDir = "E:\\1\\"
workbook = xlsxwriter.Workbook(excelDir + "整表名.xlsx")
worksheet = workbook.add_worksheet('工作簿')

content = "haha"
worksheet.write_row('A1', content)
worksheet.write_column('B2', content)

worksheet.write_column('C2', content)

workbook.close()