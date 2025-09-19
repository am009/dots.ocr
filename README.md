
# dots.ocr API

本Fork提供Docker容器重新封装的API，支持20系等旧的 Turing GPU，以float32格式运行。测试机型：单2080ti 22GB。

30系及以上显卡会默认使用bfloat，显存需求应该会减半（6-7GB显存），如果有问题可以设置环境变量`TORCH_DTYPE=float32`。我没有30系及以上显卡测试，有问题可以提issue告知。见ocr_api_server.py的第32行。

Docker一键运行命令：（API端口为5000）API使用文档见API_Documentation.md。

```bash
docker run --name dots-ocr-container -d --runtime=nvidia --gpus=all docker.io/am009/dots.ocr:latest
```

配套静态PDF识别的网页界面：http://tool.latexdiff.cn  https://tool.latexdiff.cn https://github.com/am009/LLM-online-tool

节约显存的同时，方便随时使用的方案：https://gist.github.com/am009/78989aa6597245c9444b9253e920ff64

### 本地运行

1. 配置环境。可以先尝试[这里](https://github.com/rednote-hilab/dots.ocr/issues/154#issuecomment-3233204618)的代码，确保能跑起来。如果显存不足可以把图片再缩放小一点。
1. 下载模型到`weights/DotsOCR`文件夹。
  - 如果是从ModelScope下载，注意根据[这里](https://modelscope.cn/models/rednote-hilab/dots.ocr/feedback/prDetail/44141)修改sdpa相关注意力的实现。Huggingface的修改已经被合并进去了。
  - 模型文件中的`config.json`里面的`attn_implementation`改成`sdpa`
  - 将模型文件中的`modeling_dots_vision.py`里面的bf16=True改成False。

参考：
- https://github.com/rednote-hilab/dots.ocr/issues/154#issuecomment-3233204618

提升了基于"attn_implementation": "sdpa"的显存使用效率。现在显存占用应该只会在12GB多一点，大图片也不会出现暴涨的情况了。见：https://huggingface.co/rednote-hilab/dots.ocr/discussions/27/files 或者 https://modelscope.cn/models/rednote-hilab/dots.ocr/feedback/prDetail/44141 Huggingface的修改已经被合并进去了。
