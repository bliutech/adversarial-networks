import numpy as np

import torch
from torchvision import transforms

from utils.data import CustomCIFAR

from attacks.fgsm import FGSMTransform, ToTensor, FGSMTransformCNNLSTM

from networks.cnn import CNN
from networks.rnn import CNNLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fast gradient sign method
def fgsm(image, epsilon, data_grad):
    """Generate a perturbed image using the Fast Gradient Sign Method."""
    # eta = epsilon * sign(gradient of loss w.r.t input image)
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def simple_test(test_loader, criterion, model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    losses = []
    accuracies = []
    for inputs, labels in test_loader:
        # for every batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accuracy = (torch.max(outputs, dim=1)[1] == labels).to(torch.float32).mean()
        losses.append(loss.cpu().detach().numpy())
        accuracies.append(accuracy.cpu().numpy())

    loss, accuracy = np.mean(losses), np.mean(accuracies)

    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy


def test_pre_post_fgsm(
    model, device, test_dataset_len, test_loader, epsilon, criterion
):
    """this function runs the model and compares the outputs with and without fgsm attack"""
    correct = 0
    adv_examples = []
    for images, labels in test_loader:
        # Send the data and label to the device
        images, labels = images.to(device), labels.to(device)

        # Set requires_grad attribute of tensor
        images.requires_grad = True

        # Forward pass the data through the model
        output = model(images)
        # get the index of the max log-probability
        init_pred = torch.max(output, dim=1)[1]
        loss = criterion(output, labels)

        model.zero_grad()
        loss.backward()

        # Collect gradients
        data_grad = images.grad.data

        # Call FGSM Attack
        perturbed_images = fgsm(images, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_images)

        final_pred = torch.max(output, dim=1)[
            1
        ]  # get the index of the max log-probability
        correct_idx = final_pred == labels
        correct += sum(correct_idx.to(torch.float32)).item()

        # only get pred that was right buut now wrong
        incorrect_idx = (final_pred != labels) & (init_pred == labels)

        # saving examples of perturbed images for later visualization
        if len(adv_examples) < 5:
            # Save some adv examples for visualization later
            # p is the single perturbed image, y is the correct label, initial and final and pre- and post-fgsm predictions
            for initial, final, p, y in zip(
                init_pred[incorrect_idx],
                final_pred[incorrect_idx],
                perturbed_images[incorrect_idx],
                labels[incorrect_idx],
            ):
                adv_ex = p.squeeze().detach().cpu().numpy()
                adv_examples.append((initial.item(), final.item(), y.item(), adv_ex))
                # returned adv_examples is 1 x batchsize x 4, holding items: pre-fgsm pred, post-fgsm pred, ground truth, post-fgsm image

            # Special case for saving 0 epsilon examples
            if epsilon == 0:
                for initial, final in zip(
                    init_pred[correct_idx], final_pred[correct_idx]
                ):
                    adv_ex = perturbed_images.squeeze().detach().cpu().numpy()
                    adv_examples.append(
                        (initial.item(), final.item(), final.item(), adv_ex)
                    )

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(test_dataset_len)
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_dataset_len), final_acc
        )
    )
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

EPSILONS = [0, 0.05, 0.1, 0.15, 0.2]

def compare_test(base_path):
    """given ~two~ ONE model (changed from before), compare how they do under different epsilons of fgsm attack"""
    #testing our trained model
    base_model = CNNLSTM().to(device)
    base_model.load_state_dict(torch.load(base_path))

    criterion = torch.nn.CrossEntropyLoss()

    accuracies = []
    losses = []
    for e in EPSILONS:
        transform_fgsm = transforms.Compose([
                ToTensor(),
                FGSMTransform(epsilon=e) #epsilon
            ])
        fgsm_test = CustomCIFAR(root='./data', train=False, transform=transform_fgsm)
        fgsm_loader = torch.utils.data.DataLoader(dataset=fgsm_test, batch_size=64)
        l, a = simple_test(test_loader=fgsm_loader, criterion=criterion, model=base_model)
        accuracies.append(a)
        losses.append(l)
    print("\tAccuracies:", accuracies)
    print("\tLosses:", losses)
    return accuracies, losses

EPSILONS = [0, 0.005, 0.01, 0.015, 0.2]

def compare_test_cnn_lstm(base_path, eps=EPSILONS):
    """given ~two~ ONE model (changed from before), compare how they do under different epsilons of fgsm attack"""
    #testing our trained model
    base_model = CNNLSTM().to(device)
    base_model.load_state_dict(torch.load(base_path))
    base_model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    accuracies = []
    losses = []
    for e in eps:
        transform_fgsm = transforms.Compose([
                ToTensor(),
                FGSMTransformCNNLSTM(epsilon=e) #epsilon
            ])
        fgsm_test = CustomCIFAR(root='./data', train=False, transform=transform_fgsm)
        fgsm_loader = torch.utils.data.DataLoader(dataset=fgsm_test, batch_size=64)
        l, a = simple_test(test_loader=fgsm_loader, criterion=criterion, model=base_model)
        accuracies.append(a)
        losses.append(l)
    print("\tAccuracies:", accuracies)
    print("\tLosses:", losses)
    return accuracies, losses
