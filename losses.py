import torch

def info_max_loss(softmax_out):
	# input should be already softmaxed!
	# input shape: batch_size, dim
	epsilon = 1e-5
	entropy_loss = -softmax_out * torch.log(softmax_out + 1e-5)
	entropy_loss = torch.mean(torch.sum(entropy_loss, dim=1))
	mean_softmax = torch.mean(softmax_out, dim=0)
	gentropy_loss = torch.sum(-mean_softmax * torch.log(mean_softmax + epsilon))
	im_loss = entropy_loss - gentropy_loss

	return im_loss
