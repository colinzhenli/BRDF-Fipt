% *************************************************************************
% * Copyright 2015 University of Bonn
% *
% * authors:
% *  - Sebastian Merzbach <merzbach@cs.uni-bonn.de>
% *
% * last modification date: 2015-07-10
% *
% * This file is part of btflib.
% *
% * btflib is free software: you can redistribute it and/or modify it under
% * the terms of the GNU Lesser General Public License as published by the
% * Free Software Foundation, either version 3 of the License, or (at your
% * option) any later version.
% *
% * btflib is distributed in the hope that it will be useful, but WITHOUT
% * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
% * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
% * License for more details.
% *
% * You should have received a copy of the GNU Lesser General Public
% * License along with btflib.  If not, see <http://www.gnu.org/licenses/>.
% *
% *************************************************************************
%
% Write uncompressed bidirectional image files (BDI) in Bonn University's binary
% format. A BDI file essentially stores the full BTF tensor in an ABRDF-wise
% manner, along with meta data like the bidirectional sampling. Multiple ABRDFs
% are put together into chunks. A chunk stores the ARBDFs corresponding to
% several scan lines in texture space.
% This function can in theory be used to write BDIs from non-BDI (i.e.
% compressed) BTFs, as long as those provide a decode_abrdf method.
function write_ubo_bdi(obj, fid)
    obj.write_bdi_header(fid);
    obj.write_bdi_chunks(fid);
end
