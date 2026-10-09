"""
Mini GPT — PyTorch implementation exercise.

ALLOWED PyTorch APIs:
    torch tensor operations, torch.nn.Parameter, torch.nn.Linear, torch.nn.Embedding,
    torch.nn.Module, torch.autograd, torch.optim.

BANNED PyTorch APIs (you must implement these yourself):
    torch.nn.LayerNorm, torch.nn.functional.layer_norm,
    torch.nn.MultiheadAttention, torch.nn.functional.scaled_dot_product_attention,
    torch.nn.Transformer / nn.TransformerEncoderLayer / nn.TransformerDecoderLayer,
    torch.nn.functional.softmax, torch.softmax, Tensor.softmax,
    torch.nn.functional.cross_entropy, torch.nn.functional.log_softmax,
    torch.nn.CrossEntropyLoss.

Gradients are handled entirely by autograd — you never write a backward pass. Every
forward pass you write must therefore stay differentiable: build outputs from tensor
operations on the inputs, and never call .detach(), .item(), .numpy(), or wrap
anything in torch.no_grad() except where a docstring explicitly says so.

All tensors are float32 unless stated otherwise, except `token_ids`, which is
torch.long. Do not hardcode dtypes or devices inside forward passes: derive them from
the incoming tensors, so the same code runs unchanged in float64.
"""

import torch
import torch.nn as nn


import csv
from data_pipeline import SimpleTokenizer, clean_text

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        """
        Token and positional embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of embedding vectors.
            max_seq_len (int): Maximum sequence length supported by positional embeddings.

        Attributes:
            token_embed (nn.Embedding): Token embedding table. Weight shape: (vocab_size, embed_dim)
            pos_embed (nn.Embedding): Positional embedding table. Weight shape: (max_seq_len, embed_dim)
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        """
        Computes combined token and positional embeddings for input token sequences.

        Args:
            token_ids (torch.Tensor): Token indices, dtype torch.long.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Sum of token embeddings and the positional embeddings for
                positions 0..seq_len-1, broadcast across the batch.
                Shape: (batch_size, seq_len, embed_dim)
        """
        # دریافت ابعاد ورودی
        batch_size, seq_len = token_ids.shape

        # ساخت شناسه موقعیت‌ها روی همان دستگاه ورودی
        positions = torch.arange(
            seq_len,
            device=token_ids.device
        )

        # محاسبه تعبیه توکن‌ها و موقعیت‌ها
        token_embeddings = self.token_embed(token_ids)
        position_embeddings = self.pos_embed(positions)

        # ترکیب اطلاعات توکن و موقعیت
        return token_embeddings + position_embeddings.unsqueeze(0)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """
        Layer Normalization across the feature dimension.

        Args:
            dim (int): Feature/embedding dimension to normalize.
            eps (float): Epsilon added to the variance for numerical stability.

        Attributes:
            gamma (nn.Parameter): Learnable scale, initialized to ones. Shape: (dim,)
            beta (nn.Parameter): Learnable shift, initialized to zeros. Shape: (dim,)
            eps (float): Stored epsilon value.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """
        Normalizes the last dimension of the input tensor and applies scale and shift.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (..., dim)

        Returns:
            torch.Tensor: Layer-normalized tensor with the same shape as the input.
                The mean and the biased variance are computed over the last axis only.
                Shape: (..., dim)
        """

        # محاسبه میانگین در آخرین بُعد
        mean = x.mean(dim=-1, keepdim=True)

        # محاسبه واریانس با تقسیم بر تعداد عناصر
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # نرمال‌سازی و اعمال پارامترهای قابل‌آموزش
        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gamma * normalized + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Causal Multi-Head Attention module.

        Args:
            embed_dim (int): Total dimensionality of input and output features.
            num_heads (int): Number of parallel attention heads. Must divide embed_dim.

        Attributes:
            num_heads (int): Number of attention heads.
            head_dim (int): embed_dim // num_heads.
            W_q (nn.Linear): Query projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_k (nn.Linear): Key projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_v (nn.Linear): Value projection, no bias. Weight shape: (embed_dim, embed_dim)
            W_out (nn.Linear): Output projection, no bias. Weight shape: (embed_dim, embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        for layer in (self.W_q, self.W_k, self.W_v, self.W_out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        """
        Multi-head projection, scaled dot-product attention with optional additive
        masking, and output projection. The softmax must be implemented by hand.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Additive attention mask, added to the
                attention scores before the softmax. Allowed positions hold 0.0, masked
                positions hold a large negative value (see `causal_mask`).
                Shape: (seq_len, seq_len) or broadcastable to
                (batch_size, num_heads, seq_len, seq_len). Defaults to None (no masking).

        Returns:
            torch.Tensor: Attention output after the heads are recombined and passed
                through W_out.
                Shape: (batch_size, seq_len, embed_dim)
        """

        # دریافت ابعاد ورودی
        batch_size, seq_len, embed_dim = x.shape

        # ساخت Query، Key و Value و جداسازی سرهای توجه
        q = self.W_q(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = self.W_k(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = self.W_v(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # محاسبه امتیاز توجه با مقیاس مناسب
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        # اعمال ماسک، در صورت وجود
        if mask is not None:
            mask = mask.to(device=scores.device, dtype=scores.dtype)
            scores = scores + mask

        # محاسبه دستی Softmax با پایداری عددی
        scores = scores - scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores)
        attention_weights = exp_scores / exp_scores.sum(
            dim=-1, keepdim=True
        )

        # ترکیب اطلاعات Value بر اساس وزن‌های توجه
        context = torch.matmul(attention_weights, v)

        # ادغام سرهای توجه
        context = context.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, embed_dim
        )

        # نگاشت نهایی خروجی
        return self.W_out(context)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        """
        Position-wise Feed-Forward Network (MLP).

        Args:
            embed_dim (int): Model embedding feature dimension.
            ff_dim (int): Hidden feature dimension of the expansion layer.

        Attributes:
            fc1 (nn.Linear): Expansion layer. Weight shape: (ff_dim, embed_dim), bias: (ff_dim,)
            fc2 (nn.Linear): Contraction layer. Weight shape: (embed_dim, ff_dim), bias: (embed_dim,)
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        for layer in (self.fc1, self.fc2):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Two-layer feed-forward transformation with a ReLU activation in between.

        Args:
            x (torch.Tensor): Input hidden states.
                Shape: (..., embed_dim)

        Returns:
            torch.Tensor: Transformed features, projected up to ff_dim and back down.
                Shape: (..., embed_dim)
        """

        # گسترش ویژگی‌های ورودی
        hidden = self.fc1(x)

        # اعمال تابع فعال‌سازی غیرخطی
        hidden = torch.relu(hidden)

        # بازگرداندن ویژگی‌ها به بُعد اصلی
        return self.fc2(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        """
        Pre-LayerNorm Transformer block.

        Args:
            embed_dim (int): Embedding feature dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Intermediate feed-forward layer dimension.

        Attributes:
            ln1 (LayerNorm): Normalization applied before attention.
            attn (MultiHeadAttention): Causal self-attention sub-layer.
            ln2 (LayerNorm): Normalization applied before the feed-forward network.
            ffn (FeedForward): Feed-forward sub-layer.
        """
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        """
        Passes the input through the attention and feed-forward sub-layers, each with
        pre-normalization and a residual connection.

        Args:
            x (torch.Tensor): Input representation.
                Shape: (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Additive causal mask forwarded to attention.
                Shape: (seq_len, seq_len) or broadcastable. Defaults to None.

        Returns:
            torch.Tensor: Output representation with the same shape as the input.
                Shape: (batch_size, seq_len, embed_dim)
        """

        # ذخیره ورودی برای اتصال باقی‌مانده اول
        residual = x

        # نرمال‌سازی پیش از محاسبه توجه
        normalized = self.ln1(x)

        # محاسبه توجه و افزودن اتصال باقی‌مانده
        x = residual + self.attn(normalized, mask=mask)

        # ذخیره ورودی برای اتصال باقی‌مانده دوم
        residual = x

        # نرمال‌سازی پیش از شبکه پیش‌خور
        normalized = self.ln2(x)

        # پردازش پیش‌خور و افزودن اتصال باقی‌مانده
        x = residual + self.ffn(normalized)

        return x


def causal_mask(seq_len, dtype=torch.float32, device=None):
    """
    Builds the additive causal (autoregressive) attention mask.

    Args:
        seq_len (int): Sequence length.
        dtype (torch.dtype): Data type of the returned mask. Defaults to torch.float32.
        device (torch.device, optional): Device of the returned mask. Defaults to None (CPU).

    Returns:
        torch.Tensor: Square additive mask whose entry [i, j] is 0.0 when position i is
            allowed to attend to position j (j <= i) and a large negative value
            otherwise. Use torch.finfo(dtype).min rather than a hardcoded constant so
            the mask stays valid in float64.
            Shape: (seq_len, seq_len)
    """
    # ساخت ماتریس اولیه با مقدار صفر
    mask = torch.zeros(
        (seq_len, seq_len),
        dtype=dtype,
        device=device,
    )

    # شناسایی موقعیت‌هایی که در آینده قرار دارند
    future_positions = torch.triu(
        torch.ones(
            (seq_len, seq_len),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )

    # مسدود کردن دسترسی به توکن‌های آینده
    mask = mask.masked_fill(
        future_positions,
        torch.finfo(dtype).min,
    )

    return mask

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        """
        Full MiniGPT causal language model.

        Args:
            vocab_size (int): Size of the vocabulary. Defaults to 50257.
            embed_dim (int): Hidden dimension size. Defaults to 768.
            num_heads (int): Number of attention heads. Defaults to 12.
            num_layers (int): Number of stacked Transformer blocks. Defaults to 12.
            max_seq_len (int): Maximum sequence context length. Defaults to 1024.
            ff_dim (int): Expansion dimension for the feed-forward network. Defaults to 3072.

        Attributes:
            embedding (Embedding): Joint token and positional embedding layer.
            blocks (nn.ModuleList): Stack of `num_layers` TransformerBlock modules.
            ln_f (LayerNorm): Final normalization applied before the output projection.
            vocab_size (int): Stored vocabulary size.
            embed_dim (int): Stored embedding dimension.
            max_seq_len (int): Stored maximum sequence length.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    def forward(self, token_ids):
        """
        Forward pass converting token sequences into next-token logits.

        Args:
            token_ids (torch.Tensor): Batch of token index sequences, dtype torch.long.
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Unnormalized scores over the vocabulary (logits), produced by
                weight tying with the token embedding matrix. A causal mask built from
                `seq_len` must prevent every position from attending to later positions.
                Shape: (batch_size, seq_len, vocab_size)
        """

        # بررسی شکل ورودی و محدودیت طول دنباله
        batch_size, seq_len = token_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                "طول دنباله از حداکثر طول پشتیبانی‌شده بیشتر است."
            )

        # ساخت ماسک علّی با نوع داده و دستگاه ورودی
        mask = causal_mask(
            seq_len,
            dtype=self.embedding.token_embed.weight.dtype,
            device=token_ids.device,
        )

        # تبدیل شناسه‌های توکن به بردارهای تعبیه
        x = self.embedding(token_ids)

        # عبور از تمام بلوک‌های ترنسفورمر
        for block in self.blocks:
            x = block(x, mask=mask)

        # نرمال‌سازی نهایی نمایش توکن‌ها
        x = self.ln_f(x)

        # محاسبه امتیاز واژگان با استفاده مجدد از وزن‌های تعبیه
        logits = torch.matmul(
            x,
            self.embedding.token_embed.weight.transpose(0, 1),
        )

        return logits


    def count_parameters(self):
        """
        Computes the total number of trainable parameters from the architecture itself.

        Count the token and positional embedding tables, the four attention projections,
        both feed-forward weights and biases, both LayerNorm gains and shifts of every
        block, and the final LayerNorm parameters — deriving each size from the
        hyper-parameters. Do NOT use self.parameters(); the test compares your result
        against it.

        Returns:
            int: Grand total parameter count across all components.
        """

        # تعداد پارامترهای جدول تعبیه توکن‌ها
        token_embedding_params = self.vocab_size * self.embed_dim

        # تعداد پارامترهای جدول تعبیه موقعیت‌ها
        position_embedding_params = self.max_seq_len * self.embed_dim

        # تعداد پارامترهای یک بلوک ترنسفورمر
        attention_params = 4 * self.embed_dim * self.embed_dim

        feed_forward_params = (
                self.embed_dim * self.blocks[0].ffn.fc1.out_features
                + self.blocks[0].ffn.fc1.out_features
                + self.blocks[0].ffn.fc2.in_features
                * self.blocks[0].ffn.fc2.out_features
                + self.blocks[0].ffn.fc2.out_features
        )

        layer_norm_params = 4 * self.embed_dim

        block_params = (
                attention_params
                + feed_forward_params
                + layer_norm_params
        )

        # جمع پارامترهای تمام بخش‌ها
        total_params = (
                token_embedding_params
                + position_embedding_params
                + len(self.blocks) * block_params
                + 2 * self.embed_dim
        )

        return total_params

def cross_entropy_loss(logits, targets):
    """
    Computes the average cross-entropy loss over a batch of sequences.

    Must be implemented with a numerically stable log-softmax written by hand, and must
    stay differentiable: the returned tensor is what `.backward()` is called on during
    training, so do not detach it or convert it to a Python float.

    Args:
        logits (torch.Tensor): Model output logits before softmax.
            Shape: (batch_size, seq_len, vocab_size)
        targets (torch.Tensor): Ground-truth target token indices, dtype torch.long.
            Shape: (batch_size, seq_len)

    Returns:
        torch.Tensor: Scalar (0-dimensional) loss tensor, averaged over all
            batch_size * seq_len positions.
            Shape: ()
    """

    # بررسی ابعاد ورودی‌ها
    batch_size, seq_len, vocab_size = logits.shape

    # تبدیل امتیازها به شکل دوبعدی برای محاسبه روی همه موقعیت‌ها
    logits = logits.reshape(batch_size * seq_len, vocab_size)
    targets = targets.reshape(batch_size * seq_len)

    # پایدارسازی عددی با کم کردن بیشترین امتیاز هر موقعیت
    shifted_logits = logits - logits.max(dim=-1, keepdim=True).values

    # محاسبه مجموع نمایی امتیازها
    exp_logits = torch.exp(shifted_logits)
    log_sum_exp = torch.log(
        exp_logits.sum(dim=-1, keepdim=True)
    )

    # محاسبه log-softmax به‌صورت دستی
    log_probs = shifted_logits - log_sum_exp

    # انتخاب log احتمال توکن هدف در هر موقعیت
    target_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    )

    # میانگین منفی log احتمال‌ها به‌عنوان خطای نهایی
    loss = -target_log_probs.mean()

    return loss

def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    """
    Autoregressively generates new tokens from a prompt using temperature sampling.

    Runs without gradient tracking. Sampling must use torch.multinomial, so that seeding
    with torch.manual_seed makes the output reproducible.

    Args:
        model (MiniGPT): The language model instance.
        prompt_tokens (list[int]): Initial prompt token IDs.
        max_new_tokens (int): Number of new tokens to generate. Defaults to 100.
        temperature (float): Divisor applied to the logits before the softmax. Lower
            values sharpen the distribution, higher values flatten it. Defaults to 0.8.

    Returns:
        list[int]: The prompt followed by the generated tokens, of total length
            len(prompt_tokens) + max_new_tokens. The context fed to the model at each
            step must be truncated to the model's maximum sequence length.
    """

    # بررسی ورودی‌ها
    if max_new_tokens < 0:
        raise ValueError("تعداد توکن‌های جدید نمی‌تواند منفی باشد.")

    if temperature <= 0:
        raise ValueError("دما باید بزرگ‌تر از صفر باشد.")

    if len(prompt_tokens) == 0:
        raise ValueError("پرامپت اولیه نباید خالی باشد.")

    if min(prompt_tokens) < 0 or max(prompt_tokens) >= model.vocab_size:
        raise ValueError("شناسه‌ای خارج از محدوده واژگان ورودی وجود دارد.")

    # نگهداری پرامپت و توکن‌های تولیدشده
    generated = list(prompt_tokens)

    # تولید بدون محاسبه گرادیان
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # محدود کردن زمینه به حداکثر طول مدل
            context = generated[-model.max_seq_len:]

            # تبدیل زمینه به تانسور روی دستگاه مدل
            token_ids = torch.tensor(
                [context],
                dtype=torch.long,
                device=model.embedding.token_embed.weight.device,
            )

            # محاسبه امتیازهای مدل و انتخاب آخرین موقعیت
            logits = model(token_ids)
            next_token_logits = logits[0, -1, :]

            # اعمال دما
            next_token_logits = next_token_logits / temperature

            # محاسبه دستی احتمال‌ها با پایداری عددی
            next_token_logits = (
                    next_token_logits
                    - next_token_logits.max(dim=-1, keepdim=True).values
            )
            probabilities = torch.exp(next_token_logits)
            probabilities = probabilities / probabilities.sum()

            # نمونه‌گیری از توزیع احتمال
            next_token = torch.multinomial(probabilities, num_samples=1)

            # افزودن توکن جدید به دنباله
            generated.append(next_token.item())

    return generated

def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=1500,
                   lr=3e-4, batch_size=4, tokenizer=None):
    """
    Runs an end-to-end training loop for MiniGPT on raw text.

    The text is encoded as UTF-8 bytes, so the vocabulary is the 256 possible byte
    values. Each step samples `batch_size` random windows of length seq_len + 1, uses
    the first seq_len bytes of each window as input and the last seq_len bytes as
    targets, computes the loss, backpropagates with autograd and updates every
    parameter with a torch.optim.AdamW optimizer. Print the loss every 20 steps so
    training is visible.

    Sanity check: with the defaults, the loss starts near ln(256) = 5.55 and should
    fall below 0.5 within roughly 1500 steps (about a minute on a CPU). If it plateaus
    above 3.0, something in your forward pass or loss is wrong. Note that the optimizer
    choice is part of the specification: plain SGD at this learning rate barely moves
    the loss at all.

    Args:
        text (str): Raw input text corpus used for training.
        vocab_size (int): Size of the byte vocabulary. Defaults to 256.
        embed_dim (int): Model embedding dimensionality. Defaults to 128.
        num_heads (int): Attention head count. Defaults to 4.
        num_layers (int): Transformer depth. Defaults to 4.
        seq_len (int): Training context length window, also used as the model's
            maximum sequence length. Defaults to 64.
        num_steps (int): Total gradient update steps. Defaults to 1500.
        lr (float): AdamW learning rate. Defaults to 3e-4.
        batch_size (int): Number of sequences sampled per step. Defaults to 4.

    Returns:
        MiniGPT: The trained model instance, left in eval mode.
    """

    # بررسی معتبر بودن تنظیمات آموزش
    if vocab_size < 256:
        raise ValueError("اندازه واژگان باید حداقل ۲۵۶ باشد.")

    if seq_len < 1:
        raise ValueError("طول دنباله باید حداقل یک باشد.")

    if batch_size < 1:
        raise ValueError("اندازه دسته آموزشی باید حداقل یک باشد.")

    if num_steps < 0:
        raise ValueError("تعداد مراحل آموزش نمی‌تواند منفی باشد.")

    if lr <= 0:
        raise ValueError("نرخ یادگیری باید بزرگ‌تر از صفر باشد.")

    # تبدیل متن به بایت‌های UTF-8 و شناسه‌های عددی

    # تبدیل متن به شناسه‌های عددی با توکن‌ساز مناسب
    if tokenizer is not None:
        encoded_text = tokenizer.encode(text)

        # هماهنگ کردن اندازه واژگان مدل با توکن‌ساز
        tokenizer_vocab_size = tokenizer.vocab_size()

        if vocab_size != tokenizer_vocab_size:
            vocab_size = tokenizer_vocab_size

        data = torch.tensor(encoded_text, dtype=torch.long)
    else:
        # حفظ روش قبلی برای متن‌های بدون توکن‌ساز
        encoded_text = text.encode("utf-8")
        data = torch.tensor(list(encoded_text), dtype=torch.long)


    # بررسی معتبر بودن شناسه‌های ورودی
    if data.numel() == 0:
        raise ValueError("متن آموزشی پس از توکن‌سازی خالی است.")

    if data.min().item() < 0 or data.max().item() >= vocab_size:
        raise ValueError(
            "شناسه‌های داده آموزشی با اندازه واژگان مدل سازگار نیستند."
        )

    # اطمینان از کافی بودن داده برای ساخت پنجره‌های آموزشی
    if data.numel() < seq_len + 1:
        raise ValueError(
            "متن آموزشی باید حداقل به اندازه طول دنباله به‌علاوه یک بایت باشد."
        )

    # ساخت مدل با تنظیمات درخواستی
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        ff_dim=embed_dim * 4,
    )

    # انتخاب AdamW برای به‌روزرسانی وزن‌های مدل
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # آماده‌سازی موقعیت‌های هر پنجره آموزشی
    offsets = torch.arange(seq_len)

    # قرار دادن مدل در حالت آموزش
    model.train()

    for step in range(num_steps):
        # انتخاب تصادفی نقطه شروع پنجره‌های آموزشی
        starts = torch.randint(
            low=0,
            high=data.numel() - seq_len,
            size=(batch_size,),
        )

        # ساخت ورودی‌ها و پاسخ‌های صحیح با جابه‌جایی یک بایتی
        positions = starts.unsqueeze(1) + offsets.unsqueeze(0)
        inputs = data[positions]
        targets = data[positions + 1]

        # پاک کردن گرادیان‌های مرحله قبلی
        optimizer.zero_grad()

        # پیش‌بینی توکن‌ها و محاسبه خطا
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)

        # محاسبه گرادیان‌ها و به‌روزرسانی وزن‌ها
        loss.backward()
        optimizer.step()

        # نمایش خطا هر ۲۰ مرحله
        if (step + 1) % 20 == 0:
            print(
                f"مرحله {step + 1}/{num_steps} "
                f"| خطا: {loss.item():.4f}"
            )

    # آماده‌سازی مدل برای ارزیابی یا تولید
    model.eval()

    return model



def train_mini_gpt_from_pipeline(
    documents,
    tokenizer,
    seq_len=64,
    batch_size=4,
    num_steps=200,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    lr=1e-3,
):
    """آموزش MiniGPT با خروجی واقعی پایپلاین داده."""

    from data_pipeline import (
        quality_filter,
        deduplicate,
        tokenize_corpus,
        pack_sequences,
        PreTrainingDataLoader,
    )

    # پاک‌سازی، فیلتر کیفیت و حذف اسناد تکراری
    cleaned = [clean_text(doc) for doc in documents if doc.strip()]
    filtered = [
        doc for doc in cleaned
        if quality_filter(
            doc,
            min_words=10,
            max_ratio_caps=0.5,
            max_ratio_special=0.3,
        )
    ]
    deduplicated, removed_count = deduplicate(filtered)

    if not deduplicated:
        raise ValueError(
            "هیچ سندی پس از پاک‌سازی و فیلتر کیفیت باقی نمانده است."
        )

    print(f"اسناد معتبر: {len(filtered)}")
    print(f"اسناد تکراری حذف‌شده: {removed_count}")

    # آموزش توکن‌ساز و تبدیل اسناد به شناسه‌ها
    corpus = "\n".join(deduplicated)
    tokenizer.train_bpe(corpus, num_merges=100)
    token_ids = tokenize_corpus(deduplicated, tokenizer)

    # ساخت دنباله‌هایی با یک توکن اضافه برای هدف بعدی
    sequences, masks = pack_sequences(
        token_ids, seq_len + 1, pad_id=tokenizer.pad_id
    )

    if not sequences:
        raise ValueError("پایپلاین هیچ دنباله‌ای تولید نکرد.")

    # ورودی و هدف را هم‌زمان می‌سازیم تا ارتباط آن‌ها حفظ شود
    inputs = [seq[:-1] for seq in sequences]
    targets = [seq[1:] for seq in sequences]
    target_masks = [mask[1:] for mask in masks]

    # هر ردیف ورودی، هدف و ماسک در یک ساختار مشترک نگهداری می‌شود
    from random import Random
    rng = Random(42)
    aligned = list(zip(inputs, targets, target_masks))

    vocab_size = tokenizer.vocab_size()
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        ff_dim=embed_dim * 4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # آموزش دسته‌ای با حفظ هم‌ترازی ورودی‌ها و هدف‌ها
    for step in range(num_steps):
        rng.shuffle(aligned)
        total_loss = 0.0
        total_batches = 0

        for start in range(0, len(aligned), batch_size):
            batch = aligned[start:start + batch_size]

            batch_inputs = torch.tensor(
                [item[0] for item in batch], dtype=torch.long
            )
            batch_targets = torch.tensor(
                [item[1] for item in batch], dtype=torch.long
            )
            batch_masks = torch.tensor(
                [item[2] for item in batch], dtype=torch.bool
            )

            # فقط موقعیت‌های معتبر را در محاسبه خطا وارد می‌کنیم
            valid = batch_masks & (batch_targets != tokenizer.pad_id)
            if not valid.any():
                continue

            optimizer.zero_grad()
            logits = model(batch_inputs)

            # استفاده از تابع زیان دستی تعریف‌شده در همین فایل
            loss = cross_entropy_loss(
                logits[valid].unsqueeze(1),
                batch_targets[valid].unsqueeze(1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        if (step + 1) % 20 == 0:
            average_loss = total_loss / max(total_batches, 1)
            print(
                f"مرحله {step + 1}/{num_steps} "
                f"| خطا: {average_loss:.4f}"
            )

    model.eval()
    return model

def parameter_breakdown():
    """
    Prints parameter counts for the standard GPT-2 configuration sizes.

    Returns:
        None
    """
    configs = [
        ("GPT-2 Small", 50257, 768, 12, 12, 1024, 3072),
        ("GPT-2 Medium", 50257, 1024, 16, 24, 1024, 4096),
        ("GPT-2 Large", 50257, 1280, 20, 36, 1024, 5120),
        ("GPT-2 XL", 50257, 1600, 25, 48, 1024, 6400),
    ]
    print("GPT-2 Family Parameter Counts")
    print("=" * 65)
    print(f"{'Model':<16} {'Layers':>6} {'Heads':>6} {'Dims':>6} {'Params':>14}")
    print("-" * 65)
    for name, vocab, dim, heads, layers, seq_len, ff in configs:
        token_emb = vocab * dim
        pos_emb = seq_len * dim
        per_block_attn = 4 * dim * dim
        per_block_ff = 2 * dim * ff + dim + ff
        per_block_ln = 4 * dim
        per_block = per_block_attn + per_block_ff + per_block_ln
        final_ln = 2 * dim
        total = token_emb + pos_emb + layers * per_block + final_ln
        print(f"{name:<16} {layers:>6} {heads:>6} {dim:>6} {total:>14,}")
    print()


def memory_estimate():
    """
    Prints theoretical FP16 inference memory consumption for several modern models.

    Returns:
        None
    """
    print("Memory Requirements for Inference (FP16)")
    print("=" * 65)
    models = [
        ("GPT-2 Small (124M)", 124e6, 12, 12, 64, 1024),
        ("Llama 3 8B", 8e9, 32, 32, 128, 8192),
        ("Llama 3 70B", 70e9, 80, 64, 128, 8192),
        ("Llama 3 405B", 405e9, 126, 128, 128, 131072),
    ]
    print(f"{'Model':<24} {'Weights':>10} {'KV Cache':>12} {'Total':>10}")
    print("-" * 65)
    for name, params, layers, heads, head_dim, max_seq in models:
        weight_bytes = params * 2
        kv_per_token = 2 * layers * heads * head_dim * 2
        kv_full = kv_per_token * max_seq
        total = weight_bytes + kv_full

        def fmt(b):
            if b >= 1e9:
                return f"{b / 1e9:.1f} GB"
            return f"{b / 1e6:.0f} MB"

        print(f"{name:<24} {fmt(weight_bytes):>10} {fmt(kv_full):>12} {fmt(total):>10}")
    print()



if __name__ == "__main__":
    torch.manual_seed(42)

    parameter_breakdown()
    memory_estimate()

    # خواندن اطلاعات واقعی فیلم‌ها از فایل CSV
    csv_path = "tmdb_5000_movies.csv"
    documents = []

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            title = row.get("title", "")
            tagline = row.get("tagline", "")
            overview = row.get("overview", "")

            text = clean_text(
                f"Title: {title}. Tagline: {tagline}. "
                f"Overview: {overview}"
            )

            if title.strip() and overview.strip() and len(text) >= 50:
                documents.append(text)

            # برای اجرای آزمایشی، حجم داده را محدود می‌کنیم.
            if len(documents) >= 500:
                break

    if not documents:
        raise RuntimeError("هیچ اطلاعات معتبری از فایل TMDB خوانده نشد.")

    corpus = "\n".join(documents)

    print(f"\nتعداد فیلم‌های آماده‌شده: {len(documents)}")
    print("آموزش توکن‌ساز روی اطلاعات فیلم‌ها...")

    tokenizer = SimpleTokenizer()

    print("\nشروع آموزش MiniGPT با پایپلاین داده")
    print("=" * 65)

    model = train_mini_gpt_from_pipeline(
        documents=documents,
        tokenizer=tokenizer,
        seq_len=64,
        batch_size=4,
        num_steps=40,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
    )

    prompt_text = "Title:"
    prompt = tokenizer.encode(prompt_text)

    print(f"\nپرامپت: {prompt_text}")
    print("تولید متن...")

    output_tokens = generate(
        model,
        prompt,
        max_new_tokens=80,
        temperature=0.6,
    )

    print("متن تولیدشده:")
    print(tokenizer.decode(output_tokens))

    prompt_text = "Title:"
    prompt = tokenizer.encode(prompt_text)

    print(f"\nپرامپت: {prompt_text}")
    print("تولید متن...")
    output_tokens = generate(
        model,
        prompt,
        max_new_tokens=80,
        temperature=0.6,
    )

    print("متن تولیدشده:")
    print(tokenizer.decode(output_tokens))