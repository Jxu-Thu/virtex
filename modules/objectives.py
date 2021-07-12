import torch.nn.functional as F
import torch

def image_contrastive(pl_module, batch):
    image_view_1, image_view_2 = batch["image"]
    logits_image, labels_image = pl_module.Image_Moco(image_view_1, image_view_2,mode = "intra")
    image_loss = F.cross_entropy(logits_image, labels_image)

    phase = "train" if pl_module.training else "val"
    _image_loss = getattr(pl_module, f"{phase}_IC_loss")(image_loss)
    acc_1 = getattr(pl_module, f"{phase}_IC_acc1")(logits_image, labels_image)
    acc_5 = getattr(pl_module, f"{phase}_IC_acc5")(logits_image, labels_image)
    pl_module.log(f"image/{phase}/loss", _image_loss)
    pl_module.log(f"image/{phase}/acc_1", acc_1)
    pl_module.log(f"image/{phase}/acc_5", acc_5)
    return {"IC_loss":image_loss}

def text_contrastive(pl_module, batch):
    text_ids_1, text_ids_2 = batch["text_ids"]
    text_mask_1, text_mask_2 = batch["text_masks"]
    logits_text, labels_text = pl_module.Text_Moco(text_ids_1, text_ids_2,text_mask_1, text_mask_2,mode = "intra")
    text_loss = F.cross_entropy(logits_text, labels_text)

    phase = "train" if pl_module.training else "val"
    _text_loss = getattr(pl_module, f"{phase}_TC_loss")(text_loss)
    acc_1 = getattr(pl_module, f"{phase}_TC_acc1")(logits_text, labels_text)
    acc_5 = getattr(pl_module, f"{phase}_TC_acc5")(logits_text, labels_text)
    pl_module.log(f"text/{phase}/loss", _text_loss)
    pl_module.log(f"text/{phase}/acc_1", acc_1)
    pl_module.log(f"text/{phase}/acc_5", acc_5)
    return {"TC_loss":text_loss}


def mixed_contrastive(pl_module, batch):
    """
    cross-modal contrastive learning. 
    """
    image_view_1, image_view_2 = batch["image"]
    text_mask_1, text_mask_2 = batch["text_masks"]
    text_ids_1, text_ids_2 = batch["text_ids"]
    #[b,dim]
    image_emd_1 = pl_module.Image_Moco(im_q=image_view_1, mode = "q_inter")
    image_emd_2 = pl_module.Image_Moco(im_k=image_view_2, mode = "k_inter")
    text_emd_1 = pl_module.Text_Moco(im_q=text_ids_1,text_mask_q=text_mask_1,mode = "q_inter")
    text_emd_2 = pl_module.Text_Moco(im_k=text_ids_1,text_mask_k=text_mask_1,mode = "k_inter")
    #queue:[dim,K],要得到一个[b,K]的张量，每一个元素为relu(alpha-)
    #[b,1]
    l_pos = torch.einsum('bd,bd->b', [image_emd_1, text_emd_2]).unsqueeze(-1)
    #[b,k]
    l_neg = torch.einsum('bd,dk->bk', [image_emd_1, pl_module.Text_Moco.queue_inter.clone().detach()])
    loss_image2text = torch.mean(torch.sum(F.relu(pl_module.hparams.alpha-l_pos+l_neg),dim=1))
    l_pos = torch.einsum('bd,bd->b', [text_emd_1, image_emd_2]).unsqueeze(-1)
    #[b,k]
    l_neg = torch.einsum('bd,dk->bk', [text_emd_1, pl_module.Image_Moco.queue_inter.clone().detach()])
    loss_text2image = torch.mean(torch.sum(F.relu(pl_module.hparams.alpha-l_pos+l_neg),dim=1))
    pl_module.Text_Moco._dequeue_and_enqueue(text_emd_2, mode = "inter")
    pl_module.Image_Moco._dequeue_and_enqueue(image_emd_2, mode = "inter")
    
    phase = "train" if pl_module.training else "val"
    _loss_text2image = getattr(pl_module, f"{phase}_T2IC_loss")(loss_text2image)
    _loss_image2text = getattr(pl_module, f"{phase}_I2TC_loss")(loss_image2text)
    pl_module.log(f"mix/{phase}/T2IC_loss", _loss_text2image)
    pl_module.log(f"mix/{phase}/I2TC_loss", _loss_image2text)
    return {"T2IC_loss":loss_text2image, "I2TC_loss":loss_image2text}
    
