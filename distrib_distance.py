import torch



def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def rand_projections_diff_priv(dim, num_projections=1000, sigma_proj=1):
    projections = torch.randn((num_projections, dim))*sigma_proj
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))

    return projections

def make_sample_size_equal(first_samples,second_samples):
    nb_samples_1 = first_samples.shape[0]
    nb_samples_2 = second_samples.shape[0]
    if nb_samples_1 < nb_samples_2:
        second_samples = second_samples[:nb_samples_1]
    elif nb_samples_1 > nb_samples_2:
        first_samples = first_samples[:nb_samples_2]
    return first_samples, second_samples

def sliced_wasserstein_distance(first_samples,
                                second_samples,
                                num_projections=1000,
                                p=1,
                                device='cuda'):
                

    first_samples, second_samples = make_sample_size_equal(first_samples, second_samples)
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = (second_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=1), 1. / p) # averaging the sorted distance
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)

def sliced_wasserstein_distance_diff_priv(first_samples,
                                second_samples,
                                num_projections=1000,
                                p=1,                                
                                device='cuda',
                                sigma_proj=1,
                                sigma_noise=0
                                ):
    # first samples are the data to protect
    # second samples are the data_fake
    
    first_samples, second_samples = make_sample_size_equal(first_samples, second_samples)

    dim = second_samples.size(1)
    nb_sample = second_samples.size(0)
    projections = rand_projections_diff_priv(dim, num_projections,sigma_proj)
    projections = projections.to(device)
    noise = torch.randn((nb_sample,num_projections))*sigma_noise
    noise = noise.to(device)
    noise2 = torch.randn((nb_sample,num_projections))*sigma_noise
    noise2 = noise2.to(device)    
    first_projections = first_samples.matmul(projections.transpose(0, 1)) + noise 
    second_projections = (second_samples.matmul(projections.transpose(0, 1))) + noise2 
    wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=1), 1. / p) # averaging the sorted distance
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)  # averaging over the random direction
