import torch
from torch import nn
from mask_strategy import Mask


class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # Get vocab size for each feature
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, feature_seq):
        poi_emb = self.poi_embed(feature_seq[0])
        cat_emb = self.cat_embed(feature_seq[1])
        user_emb = self.user_embed(feature_seq[2])
        hour_emb = self.hour_embed(feature_seq[3])
        day_emb = self.day_embed(feature_seq[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            embedding_layer,  # CheckInEmbedding
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)
        out = self.dropout(embedding)

        for layer in self.layers:
            out = layer(out, out, out)
        return out


class Attention(nn.Module):
    def __init__(
            self,
            q_dim,
            k_dim,
    ):
        super().__init__()
        self.expansion = nn.Linear(q_dim, k_dim)

    def forward(self, query, key, value):
        q = self.expansion(query)
        weight = torch.softmax(torch.inner(q, key), dim=0)
        weight = torch.unsqueeze(weight, 1)
        out = torch.sum(torch.mul(value, weight), 0)
        return out


class MaskedLM(nn.Module):
    def __init__(self, input_size, vocab_size):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.poi_linear = nn.Linear(input_size, vocab_size["POI"])
        self.cat_linear = nn.Linear(input_size, vocab_size["cat"])
        self.hour_linear = nn.Linear(input_size, vocab_size["hour"])

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, masked_output, masked_target):
        """calculate
        Args:
            masked_output ([type]): [description]
            masked_target ([type]): [description]
        Returns:
            [type]: [description]
        """
        poi_out = self.poi_linear(self.dropout(masked_output))
        poi_loss = self.loss_func(poi_out, masked_target[0])

        cat_out = self.cat_linear(self.dropout(masked_output))
        cat_loss = self.loss_func(cat_out, masked_target[1])

        hour_out = self.hour_linear(self.dropout(masked_output))
        hour_loss = self.loss_func(hour_out, masked_target[2])

        aux_loss = poi_loss + cat_loss + hour_loss
        return aux_loss


class RTP_CM(nn.Module):
    def __init__(
            self,
            vocab_size,
            area_code_embed_size,
            area_proportion=0.2,
            feature_embed_size=40,
            transformer_layers=1,
            transformer_heads=1,
            forward_expansion=2,
            dropout_proportion=0.1,
            back_step=2,  # future step
            mask_strategy=Mask.Simple,
            mask_proportion=0.4,
            device='cuda:0',
    ):
        super().__init__()
        self.device = device

        self.vocab_size = vocab_size
        self.vocab_size_by_num = [vocab_size["POI"], vocab_size["cat"], vocab_size["user"],
                                  vocab_size["hour"], vocab_size["day"]]

        self.total_embed_size = feature_embed_size * 5
        self.back_step = back_step

        self.mask_strategy = mask_strategy
        self.mask_proportion = mask_proportion

        self.mask_strategy_mlp = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Softmax(dim=0)
        )

        # Area auxiliary tasks
        self.area_proportion = area_proportion
        self.area_lengths = area_code_embed_size

        self.area_0_embedding = nn.Embedding(area_code_embed_size['0'], self.total_embed_size)
        self.area_1_embedding = nn.Embedding(area_code_embed_size['1'], self.total_embed_size)
        self.area_2_embedding = nn.Embedding(area_code_embed_size['2'], self.total_embed_size)
        self.area_3_embedding = nn.Embedding(area_code_embed_size['3'], self.total_embed_size)
        self.area_4_embedding = nn.Embedding(area_code_embed_size['4'], self.total_embed_size)

        self.area_dropout = nn.Dropout(0.5)
        self.area_0_dense = nn.Linear(2 * self.total_embed_size, area_code_embed_size['0'])
        self.area_1_dense = nn.Linear(2 * self.total_embed_size, area_code_embed_size['1'])
        self.area_2_dense = nn.Linear(2 * self.total_embed_size, area_code_embed_size['2'])
        self.area_3_dense = nn.Linear(2 * self.total_embed_size, area_code_embed_size['3'])
        self.area_4_dense = nn.Linear(2 * self.total_embed_size, area_code_embed_size['4'])

        # Long, short encoders
        self.embedding = CheckInEmbedding(
            feature_embed_size,
            vocab_size
        )

        self.long_term_encoder = Encoder(
            self.embedding,
            self.total_embed_size,
            transformer_layers,
            transformer_heads,
            forward_expansion,
            dropout_proportion,
        )

        self.short_term_encoder = Encoder(
            self.embedding,
            self.total_embed_size,
            transformer_layers,
            transformer_heads,
            forward_expansion,
            dropout_proportion,
        )

        if self.mask_strategy == Mask.Prediction:
            self.aux_loss = MaskedLM(
                input_size=self.total_embed_size,
                vocab_size=vocab_size
            )

        self.intra_seq_attention = Attention(
            q_dim=feature_embed_size,
            k_dim=self.total_embed_size
        )

        self.inter_seq_attention = Attention(
            q_dim=feature_embed_size,
            k_dim=self.total_embed_size
        )
        self.final_attention = Attention(
            q_dim=feature_embed_size,
            k_dim=self.total_embed_size
        )

        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_proportion),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))

        self.loss_func = nn.CrossEntropyLoss()

    def simple_mask(self, input, mask_prop=0.1):
        """
        Random mask some check-ins in long-term sequence

        Args:
            input ([[long-term sequence]]): long-term sequences in a sample
            mask_prop: mask proportion

        Returns:
            sample_list (): input random masked
        """
        sample_list = []
        for trj in input:  # each long-term trajectory
            feature_trj, areas_trj = trj[0], trj[1]
            trj_len = len(feature_trj[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(trj_len)).int()
            masked_index = torch.randperm(trj_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            # mask long-term trajectory
            feature_trj[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_trj[1, masked_index] = self.vocab_size["cat"]  # mask cat
            # 2, user is not suitable for mask
            feature_trj[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_trj[4, masked_index] = self.vocab_size["day"]  # mask day

            sample_list.append((feature_trj, areas_trj))

        return sample_list

    def living_mask(self, input, judge_level=5, living_mask_prop=0.2, non_living_mask_prop=0.4):
        sample_list = []
        area_indexes = []

        # Traverse long-term trajectories and identify the most common areas - the living area
        for trj in input:  # each long-term trajectory
            areas_trj = trj[1]
            area_indexes_at_judge_level = areas_trj[judge_level - 4]  # 4 是选用的 geohash 的最低级别 level
            area_indexes.append(area_indexes_at_judge_level)
        area_indexes = torch.cat(area_indexes, dim=0)
        living_area_index = torch.bincount(area_indexes).argmax().item()

        for trj in input:  # each long-term trajectory
            feature_trj, areas_trj = trj[0], trj[1]
            trj_len = len(feature_trj[0])
            non_living_check_in_indexes = torch.empty(0, dtype=torch.int64)
            living_check_in_indexes = torch.empty(0, dtype=torch.int64)
            for i in range(trj_len):
                area_indexes_at_judge_level = areas_trj[judge_level - 4][i]
                if area_indexes_at_judge_level != living_area_index:
                    non_living_check_in_indexes = torch.cat((non_living_check_in_indexes, torch.tensor([i])))
                else:
                    living_check_in_indexes = torch.cat((living_check_in_indexes, torch.tensor([i])))

            non_living_len = len(non_living_check_in_indexes)
            if non_living_len <= 1:
                non_living_mask_index = torch.empty(0, dtype=torch.int64)
            else:
                non_living_mask_count = torch.ceil(non_living_mask_prop * torch.tensor(non_living_len)).int()
                non_living_mask_index = torch.randperm(non_living_len - 1) + torch.tensor(1)
                non_living_mask_index = non_living_mask_index[:non_living_mask_count]

            living_len = len(living_check_in_indexes)
            if living_len <= 1:
                living_mask_index = torch.empty(0, dtype=torch.int64)
            else:
                living_mask_count = torch.ceil(living_mask_prop * torch.tensor(living_len)).int()
                living_mask_index = torch.randperm(living_len - 1) + torch.tensor(1)
                living_mask_index = living_mask_index[:living_mask_count]

            mask_index = torch.cat((torch.index_select(non_living_check_in_indexes, 0, non_living_mask_index),
                                    torch.index_select(living_check_in_indexes, 0, living_mask_index)),
                                   dim=0)

            # mask long-term trajectory
            feature_trj[0, mask_index] = self.vocab_size["POI"]  # mask POI
            feature_trj[1, mask_index] = self.vocab_size["cat"]  # mask cat
            # 2, user is not suitable for mask
            feature_trj[3, mask_index] = self.vocab_size["hour"]  # mask hour
            feature_trj[4, mask_index] = self.vocab_size["day"]  # mask day

            sample_list.append((feature_trj, areas_trj))

        return sample_list

    def prediction_mask(self, input, mask_prop=0.1):
        """generate mask (index) for long-term sequence for auxiliary training
        Args:
            input ([[long-term sequence]]): long-term sequences in a sample
        Returns:
            index_list ():
            sample_list ():
            target_list ():
        """
        index_list, sample_list = [], []
        poi_target_list, cat_target_list, user_target_list, hour_target_list, day_target_list = [], [], [], [], []
        for trj in input:  # each long-term trajectory
            feature_trj, areas_trj = trj[0], trj[1]
            trj_len = len(feature_trj[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(trj_len)).int()
            masked_index = torch.randperm(trj_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index
            index_list.append(masked_index)

            # record masked true values
            poi_target_list.append(feature_trj[0, masked_index])
            cat_target_list.append(feature_trj[1, masked_index])
            hour_target_list.append(feature_trj[3, masked_index])

            # mask long-term trajectory
            feature_trj[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_trj[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_trj[3, masked_index] = self.vocab_size["hour"]  # mask hour

            sample_list.append((feature_trj, areas_trj))

        target_list = (
            torch.cat(poi_target_list, dim=0),
            torch.cat(cat_target_list, dim=0),
            torch.cat(hour_target_list, dim=0),
        )

        return index_list, sample_list, target_list

    def forward(self, sample):
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]

        short_term_features = short_term_sequence[0][:, :-self.back_step - 1]

        user_id = short_term_sequence[0][2, 0]
        target = short_term_sequence[0][0, -self.back_step - 1]

        # Random mask long-term trajectories
        if self.mask_strategy == Mask.Simple:
            long_term_sequences = self.simple_mask(long_term_sequences, self.mask_proportion)
        elif self.mask_strategy == Mask.Living:
            long_term_sequences = self.living_mask(long_term_sequences)
        elif self.mask_strategy == Mask.Auto:
            long_term_length = sum([len(seq[0]) for seq in long_term_sequences])
            long_term_days = len(long_term_sequences)
            mlp_output = self.mask_strategy_mlp(
                torch.cuda.FloatTensor([long_term_length, long_term_days], device=self.device))
            selected_strategy = torch.argmax(mlp_output).item()
            if selected_strategy == 0:
                long_term_sequences = self.simple_mask(long_term_sequences, self.mask_proportion)
            elif selected_strategy == 1:
                long_term_sequences = self.living_mask(long_term_sequences)
        elif self.mask_strategy == Mask.AutoTrainable:
            long_term_length = sum([len(seq[0]) for seq in long_term_sequences])
            long_term_days = len(long_term_sequences)
            strategy_probs = self.mask_strategy_mlp(
                torch.cuda.FloatTensor([long_term_length, long_term_days], device=self.device))
            strategy_dist = torch.distributions.Categorical(strategy_probs)
            selected_strategy = strategy_dist.sample()
            strategy_log_prob = strategy_dist.log_prob(selected_strategy)
            if selected_strategy.item() == 0:
                long_term_sequences = self.simple_mask(long_term_sequences)
            else:
                long_term_sequences = self.living_mask(long_term_sequences)
        elif self.mask_strategy == Mask.Prediction:
            mask_index, long_term_sequences, masked_targets = self.prediction_mask(long_term_sequences)
        else:
            raise NotImplementedError()

        # Long-term encode
        long_term_out = []
        for seq in long_term_sequences:
            output = self.long_term_encoder(feature_seq=seq[0])
            long_term_out.append(output)
        long_term_catted = torch.cat(long_term_out, dim=0)

        # Short-term encode
        short_term = self.short_term_encoder(feature_seq=short_term_features)

        # Final output
        h = torch.cat((short_term, long_term_catted))  # concat short and long

        user_embed = self.embedding.user_embed(user_id)

        final_att = self.final_attention(user_embed, h, h)

        output = self.out_linear(final_att)

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)

        loss = self.loss_func(pred, label)

        # Area auxiliary task
        if self.area_proportion > 0:
            short_term_hidden_state = self.area_dropout(short_term[-1])

            area_0 = short_term_sequence[1][0, -self.back_step - 2]
            area_1 = short_term_sequence[1][1, -self.back_step - 2]
            area_2 = short_term_sequence[1][2, -self.back_step - 2]
            area_3 = short_term_sequence[1][3, -self.back_step - 2]
            area_4 = short_term_sequence[1][4, -self.back_step - 2]

            area_0_embed = self.area_dropout(self.area_0_embedding(area_0))
            area_1_embed = self.area_dropout(self.area_1_embedding(area_1))
            area_2_embed = self.area_dropout(self.area_2_embedding(area_2))
            area_3_embed = self.area_dropout(self.area_3_embedding(area_3))
            area_4_embed = self.area_dropout(self.area_4_embedding(area_4))

            area_0_target = short_term_sequence[1][0, -self.back_step - 1]
            area_1_target = short_term_sequence[1][1, -self.back_step - 1]
            area_2_target = short_term_sequence[1][2, -self.back_step - 1]
            area_3_target = short_term_sequence[1][3, -self.back_step - 1]
            area_4_target = short_term_sequence[1][4, -self.back_step - 1]

            area_0_logits = self.area_0_dense(torch.cat((short_term_hidden_state, area_0_embed), dim=0))
            area_1_logits = self.area_1_dense(torch.cat((short_term_hidden_state, area_1_embed), dim=0))
            area_2_logits = self.area_2_dense(torch.cat((short_term_hidden_state, area_2_embed), dim=0))
            area_3_logits = self.area_3_dense(torch.cat((short_term_hidden_state, area_3_embed), dim=0))
            area_4_logits = self.area_4_dense(torch.cat((short_term_hidden_state, area_4_embed), dim=0))

            area_0_loss = self.loss_func(torch.unsqueeze(area_0_logits, 0), torch.unsqueeze(area_0_target, 0))
            area_1_loss = self.loss_func(torch.unsqueeze(area_1_logits, 0), torch.unsqueeze(area_1_target, 0))
            area_2_loss = self.loss_func(torch.unsqueeze(area_2_logits, 0), torch.unsqueeze(area_2_target, 0))
            area_3_loss = self.loss_func(torch.unsqueeze(area_3_logits, 0), torch.unsqueeze(area_3_target, 0))
            area_4_loss = self.loss_func(torch.unsqueeze(area_4_logits, 0), torch.unsqueeze(area_4_target, 0))

            final_loss = (1 - self.area_proportion) * loss + self.area_proportion * self.area_lengths['0'] * (
                    area_0_loss / self.area_lengths['0'] + area_1_loss / self.area_lengths['1'] +
                    area_2_loss / self.area_lengths['2'] + area_3_loss / self.area_lengths['3'] +
                    area_4_loss / self.area_lengths['4'])
        else:
            final_loss = loss

        if self.mask_strategy == Mask.Prediction:
            aux_hidden = []
            for seq_index, seq_mask_id in enumerate(mask_index):
                aux_hidden.append(long_term_out[seq_index][seq_mask_id])
            aux_hidden = torch.squeeze(torch.cat(aux_hidden, dim=0))
            mask_prediction_loss = self.aux_loss(aux_hidden, masked_targets)
            final_loss += mask_prediction_loss
        elif self.mask_strategy == Mask.AutoTrainable:
            pred_probs = torch.softmax(pred, dim=1)
            pred_confidence = pred_probs[0][target]
            sorted_probs, _ = torch.sort(pred_probs[0], descending=True)
            top_n = 50
            threshold = sorted_probs[min(top_n, len(sorted_probs) - 1)]
            reward = (pred_confidence - threshold) / threshold
            reward = torch.clamp(reward, min=-1.0, max=1.0)
            reward = reward.detach()
            strategy_weight = 0.1
            strategy_loss = strategy_weight * strategy_log_prob * reward
            final_loss += strategy_loss
        else:
            pass

        return final_loss, output

    def predict(self, sample):
        _, pred_raw = self.forward(sample)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -self.back_step - 1]

        return ranking, target


if __name__ == "__main__":
    pass
