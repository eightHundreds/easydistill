

logits数据结构
```
[ // 1个提问的回答
    { // 第一个token
        token_id: 概率, // 概率最高的token
        token_id: 概率, // 第二,
        //.... 数量取决于,top_logits
    },
    {
        // 第二个token
    },
    {
        // 第三个token
    }
]
```

蒸馏需要
- instruction_path, 只有问题, 让模型回答获得它的答案和logits
- labeled_path, 问答对(instruction,output), 用于训练阶段,给学生模型
    白盒蒸馏中labeled_path的output意义:
    - **有意义**: output用于生成训练序列,通过formatting_func构建完整的input-output对
    - **掩码机制**: 使用labels != -100 创建掩码,只在输出token位置计算蒸馏损失
    - **双重损失**: (1-kd_ratio)*lm_loss + kd_ratio*distill_loss
    - **关键**: output提供正确答案序列,确保student和teacher在相同token位置对齐
- logits_path, 参考上面

