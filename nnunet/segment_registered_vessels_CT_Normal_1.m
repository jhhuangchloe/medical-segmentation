%% Hepatic vessels using 3D hysteresis (imreconstruct) — robust continuity, fewer islands
% Arteries = RED, Veins = BLUE
clear; close all; clc;

% ---- Paths ----
artPath  = '/Users/davidberry/Desktop/ARPA_H/CT/CT_Normal_Volumes_Anonymized/CT_Normal_1/vessel_seg/reg_ants_art_to_ven/arterial_reg_to_venous.nii.gz';
venPath  = '/Users/davidberry/Desktop/ARPA_H/CT/CT_Normal_Volumes_Anonymized/CT_Normal_1/vessel_seg/reg_ants_art_to_ven/venous_pad.nii.gz';
maskPath = '/Users/davidberry/Desktop/ARPA_H/CT/CT_Normal_Volumes_Anonymized/CT_Normal_1/vessel_seg/reg_ants_art_to_ven/liver.nii.gz';

% ---- Main knobs ----
targetMM = 0.80;      % isotropic voxel size (mm)
erodeMM  = 3.0;       % liver core erosion (mm) to avoid edge artifacts
marginMM = 12;        % crop margin around liver (mm)

% Multi-scale vesselness (pixels in isotropic space)
sensList = [1 2 3 4 5 6 8 10 12];

% Hysteresis percentiles over vesselness inside liver core:
% Keep p_high near your “good-looking” threshold; p_low lower for more branches
pA_high = 98.8;   pA_low = 98.0;   % arteries
pV_high = 98.8;   pV_low = 96.0;   % veins (lower pV_low -> more veins, more noise)

% Cleanup
doClose = true;
closeRadVox = 1;       % 1 voxel closing to bridge tiny gaps (safe-ish)
keepN_A = 2;            % keep largest N components after hysteresis
keepN_V = 2;

minVolMM3_A = 50;       % remove tiny 3D blobs after hysteresis (mm^3)
minVolMM3_V = 80;

%% ---- Read scaled NIfTI ----
[art, infoA] = readNiiScaled(artPath);
[ven, infoV] = readNiiScaled(venPath);
mask         = niftiread(maskPath) > 0;

sp = infoV.PixelDimensions;
fprintf('Spacing (mm): %.4f %.4f %.4f\n', sp(1), sp(2), sp(3));

%% ---- Crop to liver bbox (+ margin) ----
[artC, venC, maskC] = cropToMask(art, ven, mask, sp, marginMM);

%% ---- Resample cropped region to isotropic ----
scale = sp ./ targetMM;
newSize = max(1, round(size(venC) .* scale));

venI  = single(imresize3(venC,  newSize, 'linear'));
artI  = single(imresize3(artC,  newSize, 'linear'));
maskI = imresize3(single(maskC), newSize, 'nearest') > 0.5;

%% ---- Core mask + boundary downweight ----
r = max(1, round(erodeMM / targetMM));
maskCore = imerode(maskI, strel('sphere', r));

D = bwdist(~maskI);
w = min(D/(2*r), 1);
w(~maskI) = 0;

%% ---- Build inputs ----
V0 = venI; V0(~maskI) = 0;                 % veins: venous alone
A0 = artI; A0(~maskI) = 0;                 % arteries: arterial alone
A_sub = max(artI - venI, 0); A_sub(~maskI)=0; % arteries: subtraction too

% Robust normalize inside core, mild denoise
Vn    = imgaussfilt3(robustZ(V0,    maskCore, 8), 0.6);
An0   = imgaussfilt3(robustZ(A0,    maskCore, 8), 0.6);
AnSub = imgaussfilt3(robustZ(A_sub, maskCore, 8), 0.6);

%% ---- Vesselness ----
VV  = multiFibermetric(Vn, sensList);
VA  = max(multiFibermetric(An0, sensList), multiFibermetric(AnSub, sensList));

% Boundary downweight + core mask
VV = VV .* single(w); VV(~maskCore) = 0;
VA = VA .* single(w); VA(~maskCore) = 0;

% Debug MIPs
figure('Name','Vesselness MIPs');
subplot(1,3,1); imagesc(max(venI,[],3)); axis image off; title('Venous (anat) MIP');
subplot(1,3,2); imagesc(max(VA,[],3));  axis image off; title('Arterial vesselness MIP');
subplot(1,3,3); imagesc(max(VV,[],3));  axis image off; title('Venous vesselness MIP');

%% ---- 3D hysteresis via morphological reconstruction ----
BW_A = hysteresis_reconstruct(VA, maskCore, pA_high, pA_low);
BW_V = hysteresis_reconstruct(VV, maskCore, pV_high, pV_low);

fprintf('After hysteresis: A vox=%d | V vox=%d\n', nnz(BW_A), nnz(BW_V));

% Optional tiny gap closing
if doClose
    se = strel('sphere', closeRadVox);
    BW_A = imclose(BW_A, se);
    BW_V = imclose(BW_V, se);
end

% Remove tiny blobs by volume
voxelVol = targetMM^3;
BW_A = bwareaopen(BW_A, max(20, round(minVolMM3_A/voxelVol)), 26);
BW_V = bwareaopen(BW_V, max(20, round(minVolMM3_V/voxelVol)), 26);

% Keep largest N components (improves “tree-ness”)
BW_A = keepLargestN(BW_A, keepN_A);
BW_V = keepLargestN(BW_V, keepN_V);

% Resolve overlap
both = BW_A & BW_V;
BW_A(both) = VA(both) >= VV(both);
BW_V(both) = VV(both) >  VA(both);

%% ---- Binary MIPs ----
figure('Name',sprintf('Binary MIPs | A(%.1f->%.1f) V(%.1f->%.1f)',pA_high,pA_low,pV_high,pV_low));
subplot(1,2,1); imagesc(max(BW_A,[],3)); axis image off; title('Arteries (binary) MIP');
subplot(1,2,2); imagesc(max(BW_V,[],3)); axis image off; title('Veins (binary) MIP');

%% ---- 3D render ----
figure('Name','3D vessels (hysteresis reconstruct)');
render3D(maskI, BW_A, BW_V);

disp('Done.');
disp('More veins: decrease pV_low (e.g., 95.5 or 95.0).');
disp('Less noise: increase pV_high (99.0–99.5) OR increase minVolMM3_V OR set keepN_V=1.');

%% ---------------- Helper functions ----------------
function [img, info] = readNiiScaled(p)
    info = niftiinfo(p);
    raw = niftiread(info);
    img = double(raw);
    slope = 1.0; inter = 0.0;
    if isfield(info,'MultiplicativeScaling') && ~isempty(info.MultiplicativeScaling)
        slope = double(info.MultiplicativeScaling);
    end
    if isfield(info,'AdditiveOffset') && ~isempty(info.AdditiveOffset)
        inter = double(info.AdditiveOffset);
    end
    img = img .* slope + inter;
end

function [artC, venC, maskC] = cropToMask(art, ven, mask, sp, marginMM)
    idx = find(mask);
    [x,y,z] = ind2sub(size(mask), idx);
    mx = max(1, round(marginMM / sp(1)));
    my = max(1, round(marginMM / sp(2)));
    mz = max(1, round(marginMM / sp(3)));
    x1 = max(min(x)-mx, 1); x2 = min(max(x)+mx, size(mask,1));
    y1 = max(min(y)-my, 1); y2 = min(max(y)+my, size(mask,2));
    z1 = max(min(z)-mz, 1); z2 = min(max(z)+mz, size(mask,3));
    artC  = art(x1:x2, y1:y2, z1:z2);
    venC  = ven(x1:x2, y1:y2, z1:z2);
    maskC = mask(x1:x2, y1:y2, z1:z2);
end

function Z = robustZ(I, mask, clipMax)
    x = double(I(mask));
    x = x(isfinite(x));
    if isempty(x), Z = single(I); return; end
    med  = median(x);
    madv = median(abs(x - med)) + eps;
    Z = single((double(I) - med) ./ (1.4826*madv));
    Z(~mask) = 0;
    Z = max(min(Z, clipMax), 0);
end

function V = multiFibermetric(I, sensList)
    V = zeros(size(I), 'single');
    for s = sensList
        try
            Vs = single(fibermetric(I, 3, 'ObjectPolarity','bright', 'StructureSensitivity', s));
        catch
            Vs = single(fibermetric(I, 'ObjectPolarity','bright', 'StructureSensitivity', s));
        end
        V = max(V, Vs);
    end
end

function BW = hysteresis_reconstruct(V, maskCore, pHigh, pLow)
    vals = V(maskCore & V>0);
    if isempty(vals), BW = false(size(V)); return; end
    tH = prctile(vals, pHigh);
    tL = prctile(vals, pLow);

    marker = (V >= tH) & maskCore;  % high-confidence core
    mask   = (V >= tL) & maskCore;  % permissive candidates

    % Ensure marker subset of mask (should be true if tH >= tL)
    marker = marker & mask;

    conn = conndef(3,'maximal'); % 26-connectivity
    R = imreconstruct(uint8(marker), uint8(mask), conn);
    BW = R > 0;
end

function BW = keepLargestN(BW, N)
    CC = bwconncomp(BW, 26);
    if CC.NumObjects==0, return; end
    counts = cellfun(@numel, CC.PixelIdxList);
    [~, ord] = sort(counts,'descend');
    ord = ord(1:min(N, numel(ord)));
    BW2 = false(size(BW));
    BW2(vertcat(CC.PixelIdxList{ord})) = true;
    BW = BW2;
end

function render3D(maskI, BW_A, BW_V)
    hold on;
    pL = patch(isosurface(smooth3(single(maskI),'box',5), 0.5));
    pL.FaceColor = [0.85 0.85 0.85];
    pL.EdgeColor = 'none';
    pL.FaceAlpha = 0.08;

    if any(BW_A(:))
        pA = patch(isosurface(smooth3(single(BW_A),'box',3), 0.5));
        pA.FaceColor = [1 0 0];
        pA.EdgeColor = 'none';
        pA.FaceAlpha = 0.85;
    end

    if any(BW_V(:))
        pV = patch(isosurface(smooth3(single(BW_V),'box',3), 0.5));
        pV.FaceColor = [0 0 1];
        pV.EdgeColor = 'none';
        pV.FaceAlpha = 0.85;
    end

    daspect([1 1 1]);
    view(3); axis tight off;
    camlight headlight; lighting gouraud;
    title('Arteries (red) and veins (blue) within liver');
    hold off;
end
