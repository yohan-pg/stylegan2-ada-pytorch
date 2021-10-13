# class ZVariableInitAtMeanClamped(ZVariableInitAtMean):
#     def to_styles(self) -> Styles:
#         clamped = self.data.clamp(min=-0.1, max=0.1)
#         return self.G[0].mapping(
#             clamped,
#             None
#         )
