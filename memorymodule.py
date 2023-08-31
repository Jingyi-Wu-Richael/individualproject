class AutoMemoryModule(nn.Module):
    def __init__(self, embedding_layer, max_sentence_length, max_memory_context, embedding_size, padding_token, device='cpu'):
        super().__init__()
        self.max_sentence_length = max_sentence_length
        self.max_memory_context = max_memory_context
        self.embedding_size = embedding_size  # This should be 768 for "microsoft/DialoGPT-small"
        self.device = device
        self.padding_token = padding_token
        self.embedding = embedding_layer

        self.score_net_input_tokens = nn.Linear(self.embedding_size, 1).to(device=device)
        self.score_net_memory_context = nn.Linear(self.embedding_size, 1).to(device=device)

    def forward(self, input_tokens, memory_context):
        # ... (rest of the code remains the same)
        if memory_context is None:
            memory_context = torch.zeros(self.max_memory_context, dtype=torch.long).to(device=self.device)
            # fill memory context with padding tokens
            memory_context.fill_(self.padding_token)

        batch_size, seq_len = input_tokens.shape
        input_tokens = input_tokens.to(device=self.device)
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - seq_len),  value=self.padding_token)
        # score the padded input tokens
        input_tokens_embedding = self.embedding(padded_input_tokens).view(batch_size, seq_len, self.embedding_size).view(-1, self.embedding_size)
        input_tokens_scoring = self.score_net_input_tokens(input_tokens_embedding).squeeze().view(batch_size, seq_len)

        # score the memory context
        memory_context_embedding = self.embedding(memory_context).view(-1, self.embedding_size)
        memory_context_scoring = self.score_net_memory_context(memory_context_embedding).squeeze().view(-1, self.max_memory_context)

        # filter out the padding tokens from the padded input tokens and their scores
        padding_token_idx = torch.nonzero(padded_input_tokens.view(-1) != self.padding_token).squeeze(dim=1)
        filtered_input_tokens = padded_input_tokens.view(-1)[padding_token_idx]
        filtered_input_tokens_scoring = input_tokens_scoring.view(-1)[padding_token_idx]

        ctx_padding_token_idx = torch.nonzero(memory_context.view(-1) != self.padding_token).squeeze(dim=1)
        filtered_memory_context = memory_context.view(-1)[ctx_padding_token_idx]
        filtered_memory_context_scoring = memory_context_scoring.view(-1)[ctx_padding_token_idx]


        # combine the filtered input tokens and their scores with the memory context
        # and their scores
        combined_tokens = torch.cat((filtered_input_tokens, filtered_memory_context), dim=0)
        scores = torch.cat((filtered_input_tokens_scoring, filtered_memory_context_scoring), dim=0)

        # remove duplicate tokens and their scores
        unique_tokens, indices = torch.unique(combined_tokens, return_inverse=True)
        unique_scores = torch.full_like(unique_tokens, -1e20, dtype=scores.dtype)
        unique_scores = unique_scores.scatter(0, indices, scores)

        # sort the combined tokens and their scores by the scores
        sorted_scores, sorted_indices = torch.sort(unique_scores, descending=True)
        sorted_combined_tokens = unique_tokens[sorted_indices]

        # trim the combined tokens and their scores to the max memory context size
        trimmed_combined_tokens = sorted_combined_tokens[:self.max_memory_context]
        trimmed_scores = sorted_scores[:self.max_memory_context]

        # pad the trimmed tokens and their scores with padding tokens and -1e20 respectively
        trimmed_combined_tokens = nn.functional.pad(trimmed_combined_tokens, (0, self.max_memory_context - trimmed_combined_tokens.shape[-1]), value=self.padding_token)
        trimmed_scores = nn.functional.pad(trimmed_scores, (0, self.max_memory_context - trimmed_scores.shape[-1]), value=-1e20)
        # print("Padded input tokens shape:", padded_input_tokens.shape)
        # print("Max index in padded input tokens:", padded_input_tokens.max().item())
        # print("Memory context shape:", memory_context.shape)
        # print("Max index in memory context:", memory_context.max().item())
        # print("Trimmed combined tokens shape:", trimmed_combined_tokens.shape)
        # print("Max index in trimmed combined tokens:", trimmed_combined_tokens.max().item())

        return trimmed_combined_tokens, trimmed_scores