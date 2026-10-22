def accuracy_per_class(sbs, loader):
    totals = sbs.loader_apply(loader, sbs.correct)
    return totals[:, 0].float() / totals[:, 1].float()
