% 配合Step_two_item
classdef affine2d_norigid < affine2d
    methods
        function TF = isRigid(self)
            TF = isSimilarity(self) && abs(det(self.T)-1) < 1e-8;
        end
    end
end




