
# dots.ocr API

本Fork提供Docker容器重新封装的API，支持20系等旧的 Turing GPU，以float32格式运行。测试机型：单2080ti 22GB。API使用文档见API_Documentation.md


Docker一键运行命令：

```bash
TODO
```

节约显存的同时，方便随时使用的方案：https://gist.github.com/am009/78989aa6597245c9444b9253e920ff64


### 部分修改

1. 下载模型
1. 模型文件中的`config.json`里面的`attn_implementation`改成`sdpa`
1. 将模型文件中的`modeling_dots_vision.py`里面的bf16=True改成False。可能还要改一下`~/.cache/huggingface/modules/transformers_modules/DotsOCR/modeling_dots_vision.py`

参考：
- https://github.com/rednote-hilab/dots.ocr/issues/154#issuecomment-3233204618

提升了基于"attn_implementation": "sdpa"的显存使用效率。现在显存占用应该只会在12GB多一点，大图片也不会出现暴涨的情况了。

https://huggingface.co/rednote-hilab/dots.ocr/discussions/27/files 或者 https://modelscope.cn/models/rednote-hilab/dots.ocr/feedback/prDetail/44141
