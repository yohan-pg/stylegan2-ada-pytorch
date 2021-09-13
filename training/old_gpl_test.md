
            # if do_Gpl:
            #     gpl_on_z = False
            #     gpl_jvp = False

            #     #!gpl starts SUPER HIGH with adaconv
            #     with torch.autograd.profiler.record_function("Gpl_forward"):
            #         batch_size = gen_z.shape[0] // self.pl_batch_shrink
            #         z = gen_z[:batch_size].requires_grad_(True)

            #         if not gpl_jvp:
            #             gen_img, gen_ws = self.run_G(
            #                 z, gen_c[:batch_size], sync=sync
            #             )
            #             pl_noise = torch.randn_like(gen_img) / np.sqrt(
            #                 gen_img.shape[2] * gen_img.shape[3]
            #             )
            #             with torch.autograd.profiler.record_function(
            #                 "pl_grads"
            #             ), conv2d_gradfix.no_weight_gradients():
            #                 pl_grads = torch.autograd.grad(
            #                     outputs=[(gen_img * pl_noise).sum()],
            #                     inputs=[z if gpl_on_z else gen_ws],
            #                     create_graph=True,
            #                     only_inputs=True,
            #                 )[0]
            #             if gpl_on_z:
            #                 pl_lengths = pl_grads.square().sum(1).sqrt()
            #             else:
            #                 pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            #             pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            #             self.pl_mean.copy_(pl_mean.detach())
            #             pl_penalty = (pl_lengths - pl_mean).square()
            #         else:
            #             #* Try to measure using a jvp instead, this should avoid stepping in random image-space directions
            #             # jvp = torch.autograd.functional.jvp(
            #             #     torch.randn_like(z) / math.sqrt(z.shape[1])
            #             # )
            #             raise NotImplementedError

            #         training_stats.report("Loss/pl_penalty", pl_penalty)
            #         loss_Gpl = pl_penalty * self.pl_weight
            #         training_stats.report("Loss/G/reg", loss_Gpl)

            #     with torch.autograd.profiler.record_function("Gpl_backward"):
            #         (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()