def hook_fn(module, input, output):
    """
    Hook function to store the intermediate embeddings of nodes after each convolution.
    """
    # Store the output embeddings
    hook_fn.embeddings.append(output.clone().detach())

def register_hooks(model, hook_handles):
    """
    Register forward hooks to GIN convolution layers to capture embeddings.
    """
    hook_fn.embeddings = []  # List to store embeddings
    hook_handles.append(model.conv1.register_forward_hook(hook_fn))  # Hook after first GINConv
    hook_handles.append(model.conv2.register_forward_hook(hook_fn))  # Hook after second GINConv
    hook_handles.append(model.conv3.register_forward_hook(hook_fn))  # Hook after second GINConv
    hook_handles.append(model.fc1.register_forward_hook(hook_fn))    # Hook after third layer (or another layer)

def remove_hooks(hook_handles):
    """
    Remove all registered forward hooks.
    """
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()  # Clear the list after removing hooks