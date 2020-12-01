# TFRecord

TFRecord 是谷歌推荐的一种二进制文件格式，理论上它可以保存任何格式的信息。

### TFRecord文档结构

```
uint64 length	# 文件长度，0-
uint32 masked_crc32_of_length	# 长度校验码
byte   data[length]				# 数据，二进制
uint32 masked_crc32_of_data		# 数据校验码
```

# Protocol buffer

Protocol Buffers(也称protobuf)是Google公司出口的一种**独立于开发语言**，独立于平台的可扩展的结构化数据序列机制。**通俗点来讲它跟xml和json是一类**。是一种数据交互格式协议。