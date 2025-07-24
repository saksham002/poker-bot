from torch.utils.tensorboard import SummaryWriter

class Plotter:
    def __init__(self, player_names):
        self.num_players = len(player_names)
        self.writer = SummaryWriter('plots')
        self.player_names = player_names
        self.iter_counts = {name: 0 for name in player_names}

    def log_data(self, plot_data):
        for i, player_name in enumerate(self.player_names):
            player_plot_data = plot_data[i]
            for critic_loss, policy_loss, fold_loss, total_policy_loss, reward, money_value, policy_grad_norm, critic_grad_norm, batch_size in zip(player_plot_data["critic_losses"], player_plot_data["policy_losses"],                                                                                             player_plot_data["fold_losses"], player_plot_data["total_policy_losses"],                                                                                             player_plot_data["rewards"], player_plot_data["money_values"],                                                                                             player_plot_data["policy_grad_norms"], player_plot_data["critic_grad_norms"], player_plot_data["batch_sizes"]):
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Critic_Loss', critic_loss, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Policy_Loss', policy_loss, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Fold_Loss', fold_loss, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Total_Policy_Loss', total_policy_loss, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Reward', reward, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Money_Value', money_value, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Policy_Grad_Norm', policy_grad_norm, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Critic_Grad_Norm', critic_grad_norm, self.iter_counts[player_name])
                self.writer.add_scalar(f'{self.num_players}/{player_name}/Batch_Size', batch_size, self.iter_counts[player_name])
                self.iter_counts[player_name] += 1

    def close(self):
        self.writer.close()
