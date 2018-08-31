/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cuda/CudaBVH.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaBVH::CudaBVH(const BVH& bvh, BVHLayout layout)
:   m_layout    (layout)
{
    FW_ASSERT(layout >= 0 && layout < BVHLayout_Max);

    if (layout == BVHLayout_Compact)
    {
        createCompact(bvh,1);
        return;
    }

    if (layout == BVHLayout_Compact2)
    {
        createCompact(bvh,16);
        return;
    }

    if (layout == BVHLayout_Compact2_Idx)
    {
        createCompact(bvh, 16, true);
        return;
    }

    if (layout == BVHLayout_Bin)
    {
        createBin(bvh, 16);
        return;
    }

    if (layout == BVHLayout_Bin_Idx)
    {
        createBin(bvh, 16, true);
        return;
    }

    if (layout == BVHLayout_Quad)
    {
        createQuad(bvh, 16);
        return;
    }

    if (layout == BVHLayout_Quad_Idx)
    {
        createQuad(bvh, 16, true);
        return;
    }

    if (layout == BVHLayout_Oct)
    {
        createOct(bvh, 16);
        return;
    }

    if (layout == BVHLayout_Oct_Idx)
    {
        createOct(bvh, 16, true);
        return;
    }

    if (layout == BVHLayout_Oct_Mini)
    {
        createOctMini(bvh, 16, true);
        return;
    }

        createNodeBasic(bvh);
    createTriWoopBasic(bvh);
    createTriIndexBasic(bvh);
}

//------------------------------------------------------------------------

CudaBVH::CudaBVH(InputStream& in)
{
    in >> (S32&)m_layout >> m_nodes >> m_triWoop >> m_triIndex
       >> m_triWoop_v0 >> m_triWoop_v1 >> m_triWoop_v2
       >> m_nodes_XY >> m_nodes_ZI;
}

//------------------------------------------------------------------------

CudaBVH::~CudaBVH(void)
{
}

//------------------------------------------------------------------------

void CudaBVH::serialize(OutputStream& out)
{
    out << (S32)m_layout << m_nodes << m_triWoop << m_triIndex
        << m_triWoop_v0 << m_triWoop_v1 << m_triWoop_v2
        << m_nodes_XY << m_nodes_ZI;
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getNodeSubArray(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < 4);
    S32 size = (S32)m_nodes.getSize();

    if (m_layout == BVHLayout_SOA_AOS || m_layout == BVHLayout_SOA_SOA)
        return Vec2i((size >> 2) * idx, (size >> 2));
    return Vec2i(0, size);
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getTriWoopSubArray(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < 4);
    S32 size = (S32)m_triWoop.getSize();

    if (m_layout == BVHLayout_AOS_SOA || m_layout == BVHLayout_SOA_SOA)
        return Vec2i((size >> 2) * idx, (size >> 2));
    return Vec2i(0, size);
}

//------------------------------------------------------------------------

CudaBVH& CudaBVH::operator=(CudaBVH& other)
{
    if (&other != this)
    {
        m_layout    = other.m_layout;
        m_nodes     = other.m_nodes;
        m_triWoop   = other.m_triWoop;
        m_triIndex  = other.m_triIndex;
    }
    return *this;
}

//------------------------------------------------------------------------

void CudaBVH::createNodeBasic(const BVH& bvh)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
        int encodeIdx(void) const { return (node->isLeaf()) ? ~idx : idx; }
    };

    const BVHNode* root = bvh.getRoot();
    m_nodes.resizeDiscard((root->getSubtreeSize(BVH_STAT_NODE_COUNT) * 64 + Align - 1) & -Align);

    int nextNodeIdx = 0;
    Array<StackEntry> stack(StackEntry(root, nextNodeIdx++));
    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        const AABB* b0;
        const AABB* b1;
        int c0;
        int c1;

        // Leaf?

        if (e.node->isLeaf())
        {
            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(e.node);
            b0 = &leaf->m_bounds;
            b1 = &leaf->m_bounds;
            c0 = leaf->m_lo;
            c1 = leaf->m_hi;
        }

        // Internal node?

        else
        {
            StackEntry e0 = stack.add(StackEntry(e.node->getChildNode(0), nextNodeIdx++));
            StackEntry e1 = stack.add(StackEntry(e.node->getChildNode(1), nextNodeIdx++));
            b0 = &e0.node->m_bounds;
            b1 = &e1.node->m_bounds;
            c0 = e0.encodeIdx();
            c1 = e1.encodeIdx();
        }

        // Write entry.

        Vec4i data[] =
        {
            Vec4i(floatToBits(b0->min().x), floatToBits(b0->max().x), floatToBits(b0->min().y), floatToBits(b0->max().y)),
            Vec4i(floatToBits(b1->min().x), floatToBits(b1->max().x), floatToBits(b1->min().y), floatToBits(b1->max().y)),
            Vec4i(floatToBits(b0->min().z), floatToBits(b0->max().z), floatToBits(b1->min().z), floatToBits(b1->max().z)),
            Vec4i(c0, c1, 0, 0)
        };

        switch (m_layout)
        {
        case BVHLayout_AOS_AOS:
        case BVHLayout_AOS_SOA:
            memcpy(m_nodes.getMutablePtr(e.idx * 64), data, 64);
            break;

        case BVHLayout_SOA_AOS:
        case BVHLayout_SOA_SOA:
            for (int i = 0; i < 4; i++)
                memcpy(m_nodes.getMutablePtr(e.idx * 16 + (m_nodes.getSize() >> 2) * i), &data[i], 16);
            break;

        default:
            FW_ASSERT(false);
            break;
        }
    }
}

//------------------------------------------------------------------------

void CudaBVH::createTriWoopBasic(const BVH& bvh)
{
    const Array<S32>& tidx = bvh.getTriIndices();
    m_triWoop.resizeDiscard((tidx.getSize() * 64 + Align - 1) & -Align);

    for (int i = 0; i < tidx.getSize(); i++)
    {
        woopifyTri(bvh, i);

        switch (m_layout)
        {
        case BVHLayout_AOS_AOS:
        case BVHLayout_SOA_AOS:
            memcpy(m_triWoop.getMutablePtr(i * 64), m_woop, 48);
            break;

        case BVHLayout_AOS_SOA:
        case BVHLayout_SOA_SOA:
            for (int j = 0; j < 3; j++)
                memcpy(m_triWoop.getMutablePtr(i * 16 + (m_triWoop.getSize() >> 2) * j), &m_woop[j], 16);
            break;

        default:
            FW_ASSERT(false);
            break;
        }
    }
}

//------------------------------------------------------------------------

void CudaBVH::createTriIndexBasic(const BVH& bvh)
{
    const Array<S32>& tidx = bvh.getTriIndices();
    m_triIndex.resizeDiscard(tidx.getSize() * 4);

    for (int i = 0; i < tidx.getSize(); i++)
        *(S32*)m_triIndex.getMutablePtr(i * 4) = tidx[i];
}


int CudaBVH::peekSubTreeSize8(const BVHNode* node)
{
	// BVH collapse adapted from Michael Guthe's implementation of his paper 'Latency Considerations of Depth-first GPU Ray Tracing'.
	
    const AABB* cbox[8];
    int cidx[8];

    // Collect children.
    BVHNode*    child[8];
    int ccnt = 2;
    child[0] = node->getChildNode(0);
    child[1] = node->getChildNode(1);
    child[2] = NULL;
    child[3] = NULL;
    child[4] = NULL;
    child[5] = NULL;
    child[6] = NULL;
    child[7] = NULL;

    while (ccnt < 8)
    {
        int     s = -1;
        float   a = -1.0f;
        int     p = 0;


        for (int i = 0; i < ccnt; ++i)
        {
            if (!child[i]->isLeaf())
            {
                float ca = child[i]->getArea();
                if (ca > a)
                {
                    a = ca;
                    s = i;
                }
            }
        }

        if (s == -1) break;

        for (int i = ccnt; i > s + 1; --i)
            child[i] = child[i - 1];
        BVHNode*    sc = child[s];

        child[s] = sc->getChildNode(0);
        child[s + 1] = sc->getChildNode(1);
        ++ccnt;
    }

    return ccnt;
}


//------------------------------------------------------------------------

void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv, bool indexed)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };

    // Construct data.

    Array<Vec4i> nodeData(NULL, 4);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0));

    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[2];
        int cidx[2];

        // Process children.

        for (int i = 0; i < 2; i++)
        {
            // Inner node => push to stack.

            const BVHNode* child = e.node->getChildNode(i);
            cbox[i] = &child->m_bounds;
            if (!child->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                stack.add(StackEntry(child, nodeData.getSize()));
                nodeData.add(NULL, 4);
                continue;
            }

            // Leaf => append triangles.
            if (indexed) {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
                cidx[i] = ~triIndexData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    triIndexData.add(bvh.getTriIndices()[j]);
                }

                // Terminator.

                triIndexData.add(-1);
            }

            else {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
                cidx[i] = ~triWoopData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    woopifyTri(bvh, j);
                    if (m_woop[0].x == 0.0f)
                        m_woop[0].x = 0.0f;
                    triWoopData.add((Vec4i*)m_woop, 3);
                    triIndexData.add(bvh.getTriIndices()[j]);
                    triIndexData.add(0);
                    triIndexData.add(0);
                }

                // Terminator.

                triWoopData.add(0x80000000);
                triIndexData.add(0);
            }
        }

        // Write entry.

        Vec4i* dst = nodeData.getPtr(e.idx);
        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
        dst[3] = Vec4i(cidx[0], cidx[1], 0, 0);
    }

    if (indexed) {

        // Store triangles without duplicates.

        for (int i = 0; i < bvh.getScene()->getNumTriangles(); ++i)
        {
            woopifyTri_bypass(bvh, i);
            if (m_woop[0].x == 0.0f)
                m_woop[0].x = 0.0f;

            triWoopData.add((Vec4i*)m_woop, 3);
        }
    }

    // Write to buffers.

    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());
}


//------------------------------------------------------------------------

void CudaBVH::createBin(const BVH& bvh, int nodeOffsetSizeDiv, bool indexed)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };

    // Construct data.

    Array<Vec4i> nodeData(NULL, 4);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0));

    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[2];
        int cidx[2];

        // Process children.

        for (int i = 0; i < 2; i++)
        {
            // Inner node => push to stack.

            const BVHNode* child = e.node->getChildNode(i);
            cbox[i] = &child->m_bounds;
            if (!child->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                stack.add(StackEntry(child, nodeData.getSize()));
                nodeData.add(NULL, 4);
                continue;
            }

            // Leaf => append triangles.

            if (indexed) {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
                cidx[i] = ~triIndexData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    triIndexData.add(bvh.getTriIndices()[j]);
                }
            }
            else {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
                cidx[i] = ~triWoopData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    woopifyTri(bvh, j);
                    if (m_woop[0].x == 0.0f)
                        m_woop[0].x = 0.0f;
                    triWoopData.add((Vec4i*)m_woop, 3);
                    triIndexData.add(bvh.getTriIndices()[j]);
                    triIndexData.add(0);
                    triIndexData.add(0);
                }
            }

        }

        // Write entry.

        Vec4i* dst = nodeData.getPtr(e.idx);
        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), cidx[0], 0);
        dst[3] = Vec4i(floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z), cidx[1], 0);
    }

    // Store triangles without duplicates.
    if (indexed) {
        for (int i = 0; i < bvh.getScene()->getNumTriangles(); ++i)
        {
            woopifyTri_bypass(bvh, i);
            if (m_woop[0].x == 0.0f)
                m_woop[0].x = 0.0f;
            triWoopData.add((Vec4i*)m_woop, 3);
        }

        // Dummies.

        triWoopData.add(0x80000000);
        triIndexData.add(0);

    }

    // Write to buffers.

    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());
}


//------------------------------------------------------------------------

void CudaBVH::createQuad(const BVH& bvh, int nodeOffsetSizeDiv, bool indexed)
{
	// BVH collapse adapted from Michael Guthe's implementation of his paper 'Latency Considerations of Depth-first GPU Ray Tracing'.

    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };

    // Construct data.
    
    Array<Vec4i> nodeData(NULL, 8);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0));

    while (stack.getSize())
    {

        StackEntry e = stack.removeLast();
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[4];
        int cidx[4];

        // Collect children.
        BVHNode*    child[4];
        int ccnt = 2;
        child[0] = e.node->getChildNode(0);
        child[1] = e.node->getChildNode(1);
        child[2] = NULL;
        child[3] = NULL;

        while (ccnt<4)
        {
            int     s = -1;
            float   a = -1.0f;
            int     p = 0;
            
            for (int i = 0; i<ccnt; ++i)
            {
                if (!child[i]->isLeaf())
                {
                    float       ca = child[i]->getArea();
                    int         cp = 2; // normal
                    const int   lcnt = child[i]->getSubtreeSize(BVH_STAT_LEAF_COUNT);

                    // reduces memory, almost no effect on performance
                    if (ccnt == 2)
                    {
                        if (lcnt == 4) cp = 1;  // low (try to prevent)
                        if (lcnt == 3) cp = 3;  // high (removes a three leaf node from the tree)
                        if (lcnt == 2) cp = 4;  // high (removes a two leaf node from the tree)
                    }
                    else
                    {
                        if ((lcnt == 3) || (lcnt == 4)) cp = 1; // low (try to prevent)
                        if (lcnt == 2) cp = 3;              // high (removes a two leaf node from the tree)
                    }
                    if ((cp>p) || ((cp == p) && (ca>a)))
                    {
                        a = ca;
                        s = i;
                        p = cp;
                    }
                }
            }

            if (s == -1) break;

            for (int i = ccnt; i>s + 1; --i)
                child[i] = child[i - 1];
            BVHNode*    sc = child[s];

            child[s] = sc->getChildNode(0);
            child[s + 1] = sc->getChildNode(1);
            ++ccnt;
        }

        // Process children.

        for (int i = 0; i < ccnt; i++)
        {
            // Inner node => push to stack.

            cbox[i] = &child[i]->m_bounds;
            if (!child[i]->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                stack.add(StackEntry(child[i], nodeData.getSize()));
                nodeData.add(NULL, 8);
                continue;
            }

            if (indexed)
            {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child[i]);
                cidx[i] = ~triIndexData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    triIndexData.add(bvh.getTriIndices()[j]);
                }
            }
            else {
                // Leaf => append triangles.
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child[i]);
                cidx[i] = ~triWoopData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    woopifyTri(bvh, j);
                    if (m_woop[0].x == 0.0f)
                        m_woop[0].x = 0.0f;
                    triWoopData.add((Vec4i*)m_woop, 3);
                    triIndexData.add(bvh.getTriIndices()[j]);
                    triIndexData.add(0);
                    triIndexData.add(0);
                }
            }

        }
        for (int i = ccnt; i<4; ++i)
        {
            cbox[i] = new AABB(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f));
            cidx[i] = 0;
        }

        // Write entry.

        Vec4i* dst = nodeData.getPtr(e.idx);    
        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[2]->min().x), floatToBits(cbox[2]->max().x), floatToBits(cbox[2]->min().y), floatToBits(cbox[2]->max().y));
        dst[3] = Vec4i(floatToBits(cbox[3]->min().x), floatToBits(cbox[3]->max().x), floatToBits(cbox[3]->min().y), floatToBits(cbox[3]->max().y));
        
        dst[4] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), cidx[0], 0);
        dst[5] = Vec4i(floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z), cidx[1], 0);
        dst[6] = Vec4i(floatToBits(cbox[2]->min().z), floatToBits(cbox[2]->max().z), cidx[2], 0);
        dst[7] = Vec4i(floatToBits(cbox[3]->min().z), floatToBits(cbox[3]->max().z), cidx[3], 0);
        
        for (int i = ccnt; i<4; ++i)
        {
            delete cbox[i];
        }
    }

    if (indexed) {
        // Store all triangels without duplicates.
        for (int i = 0; i < bvh.getScene()->getNumTriangles(); ++i)
        {
            woopifyTri_bypass(bvh, i);
            if (m_woop[0].x == 0.0f)
                m_woop[0].x = 0.0f;

            triWoopData.add((Vec4i*)m_woop, 3);
        }
    }

    // Dummy.
    for (int i = 0; i < 3; ++i) {
        triWoopData.add(0x80000000);
        triIndexData.add(0);
    }
    
    // Write to buffers.
    
    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());

}


//------------------------------------------------------------------------

void CudaBVH::createOct(const BVH& bvh, int nodeOffsetSizeDiv, bool indexed)
{
	// BVH collapse adapted from Michael Guthe's implementation of his  paper 'Latency Considerations of Depth-first GPU Ray Tracing'.

    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };

    // Construct data.

    Array<Vec4i> nodeData(NULL, 16);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0));

    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[8];
        int cidx[8];

        // Collect children.
        BVHNode*    child[8];
        int ccnt = 2;
        child[0] = e.node->getChildNode(0);
        child[1] = e.node->getChildNode(1);
        child[2] = NULL;
        child[3] = NULL; 
        child[4] = NULL;
        child[5] = NULL;
        child[6] = NULL;
        child[7] = NULL;

        while (ccnt<8)
        {
            int     s = -1;
            float   a = -1.0f;
            int     p = 0;

            
            for (int i = 0; i < ccnt; ++i)
            {
                if (!child[i]->isLeaf())
                {
                    float ca = child[i]->getArea();
                    if (ca > a)
                    {
                        a = ca;
                        s = i;
                    }
                }
            }
            
            if (s == -1) break;

            for (int i = ccnt; i>s + 1; --i)
                child[i] = child[i - 1];
            BVHNode*    sc = child[s];

            child[s] = sc->getChildNode(0);
            child[s + 1] = sc->getChildNode(1);
            ++ccnt;
        }

        // Process children.

        for (int i = 0; i < ccnt; i++)
        {
            // Inner node => push to stack.

            cbox[i] = &child[i]->m_bounds;
            if (!child[i]->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                stack.add(StackEntry(child[i], nodeData.getSize()));
                nodeData.add(NULL, 16);
                continue;
            }

            if (indexed) {

                // Leaf => append triangel indices
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child[i]);
                cidx[i] = ~triIndexData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    triIndexData.add(bvh.getTriIndices()[j]);
                }
            }
            else {

                // Leaf => append triangles.
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child[i]);
                cidx[i] = ~triWoopData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    woopifyTri(bvh, j);
                    if (m_woop[0].x == 0.0f)
                        m_woop[0].x = 0.0f;
                    triWoopData.add((Vec4i*)m_woop, 3);
                    triIndexData.add(bvh.getTriIndices()[j]);
                    triIndexData.add(0);
                    triIndexData.add(0);
                }
            }

        }
        for (int i = ccnt; i<8; ++i)
        {
            cbox[i] = new AABB(Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f));
            cidx[i] = 0;
        }

        // Write entry.

        Vec4i* dst = nodeData.getPtr(e.idx);
    

        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[2]->min().x), floatToBits(cbox[2]->max().x), floatToBits(cbox[2]->min().y), floatToBits(cbox[2]->max().y));
        dst[3] = Vec4i(floatToBits(cbox[3]->min().x), floatToBits(cbox[3]->max().x), floatToBits(cbox[3]->min().y), floatToBits(cbox[3]->max().y));
        dst[4] = Vec4i(floatToBits(cbox[4]->min().x), floatToBits(cbox[4]->max().x), floatToBits(cbox[4]->min().y), floatToBits(cbox[4]->max().y));
        dst[5] = Vec4i(floatToBits(cbox[5]->min().x), floatToBits(cbox[5]->max().x), floatToBits(cbox[5]->min().y), floatToBits(cbox[5]->max().y));
        dst[6] = Vec4i(floatToBits(cbox[6]->min().x), floatToBits(cbox[6]->max().x), floatToBits(cbox[6]->min().y), floatToBits(cbox[6]->max().y));
        dst[7] = Vec4i(floatToBits(cbox[7]->min().x), floatToBits(cbox[7]->max().x), floatToBits(cbox[7]->min().y), floatToBits(cbox[7]->max().y));

        dst[8] =  Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), cidx[0], 0);
        dst[9] =  Vec4i(floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z), cidx[1], 0);
        dst[10] = Vec4i(floatToBits(cbox[2]->min().z), floatToBits(cbox[2]->max().z), cidx[2], 0);
        dst[11] = Vec4i(floatToBits(cbox[3]->min().z), floatToBits(cbox[3]->max().z), cidx[3], 0);
        dst[12] = Vec4i(floatToBits(cbox[4]->min().z), floatToBits(cbox[4]->max().z), cidx[4], 0);
        dst[13] = Vec4i(floatToBits(cbox[5]->min().z), floatToBits(cbox[5]->max().z), cidx[5], 0);
        dst[14] = Vec4i(floatToBits(cbox[6]->min().z), floatToBits(cbox[6]->max().z), cidx[6], 0);
        dst[15] = Vec4i(floatToBits(cbox[7]->min().z), floatToBits(cbox[7]->max().z), cidx[7], 0);

        for (int i = ccnt; i<8; ++i)
        {
            delete cbox[i];
        }
    }

    if (indexed) {

        // Store all triangels without duplicates.
        for (int i = 0; i < bvh.getScene()->getNumTriangles(); ++i)
        {
            woopifyTri_bypass(bvh, i);
            if (m_woop[0].x == 0.0f)
                m_woop[0].x = 0.0f;

            triWoopData.add((Vec4i*)m_woop, 3);
        }
    }

    // Dummy.
    for (int i = 0; i < 7; ++i) {
        triWoopData.add(0x80000000);
        triIndexData.add(0);
    }


    // Write to buffers.

    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());

}


//------------------------------------------------------------------------

void CudaBVH::createOctMini(const BVH& bvh, int nodeOffsetSizeDiv, bool indexed)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;
        S32             numNodes;

        StackEntry(const BVHNode* n = NULL, int i = 0, int num = -1) : node(n), idx(i), numNodes(num) {}
    };

    // Construct data.

    int nodeWidth = peekSubTreeSize8(bvh.getRoot());
    Array<Vec4i> nodeDataXY(NULL, nodeWidth);
    Array<Vec4i> nodeDataZI(NULL, nodeWidth);
    Array<Vec4f> triWoopData_v0;
    Array<Vec4f> triWoopData_v1;
    Array<Vec4f> triWoopData_v2;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0, nodeWidth));

    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[8];
        int cidx[8];

        // Collect children.
        BVHNode*    child[8];
        int ccnt = 2;
        child[0] = e.node->getChildNode(0);
        child[1] = e.node->getChildNode(1);
        child[2] = NULL;
        child[3] = NULL;
        child[4] = NULL;
        child[5] = NULL;
        child[6] = NULL;
        child[7] = NULL;

        while (ccnt<8)
        {
            int     s = -1;
            float   a = -1.0f;
            int     p = 0;


            for (int i = 0; i < ccnt; ++i)
            {
                if (!child[i]->isLeaf())
                {
                    float ca = child[i]->getArea();
                    if (ca > a)
                    {
                        a = ca;
                        s = i;
                    }
                }
            }

            if (s == -1) break;

            for (int i = ccnt; i>s + 1; --i)
                child[i] = child[i - 1];
            BVHNode*    sc = child[s];

            child[s] = sc->getChildNode(0);
            child[s + 1] = sc->getChildNode(1);
            ++ccnt;
        }

        FW_ASSERT(ccnt == e.numNodes);

        // Process children.

        for (int i = 0; i < ccnt; i++)
        {
            // Inner node => push to stack.

            cbox[i] = &child[i]->m_bounds;
            if (!child[i]->isLeaf())
            {
                cidx[i] = nodeDataXY.getNumBytes() / nodeOffsetSizeDiv;
                int nodeWidth = peekSubTreeSize8(child[i]);
                stack.add(StackEntry(child[i], nodeDataXY.getSize(), nodeWidth));
                nodeDataXY.add(NULL, nodeWidth);
                nodeDataZI.add(NULL, nodeWidth);
                continue;
            }

            // Leaf => append triangles.

            if (indexed) {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child[i]);
                cidx[i] = ~triIndexData.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    triIndexData.add(bvh.getTriIndices()[j]);
                }
            }
            else {
                const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child[i]);
                cidx[i] = ~triWoopData_v0.getSize();
                for (int j = leaf->m_lo; j < leaf->m_hi; j++)
                {
                    woopifyTri(bvh, j);
                    if (m_woop[0].x == 0.0f)
                        m_woop[0].x = 0.0f;

                    triWoopData_v0.add(m_woop[0]);
                    triWoopData_v1.add(m_woop[1]);
                    triWoopData_v2.add(m_woop[2]);
                    triIndexData.add(bvh.getTriIndices()[j]);

                }
            }

        }

        // Write entry.

        Vec4i* dstXY = nodeDataXY.getPtr(e.idx);
        Vec4i* dstZI = nodeDataZI.getPtr(e.idx);

        for (int i = 0; i < ccnt; ++i) {
            dstXY[i] = Vec4i(floatToBits(cbox[i]->min().x), floatToBits(cbox[i]->max().x), floatToBits(cbox[i]->min().y), floatToBits(cbox[i]->max().y));
            dstZI[i] = Vec4i(floatToBits(cbox[i]->min().z), floatToBits(cbox[i]->max().z), cidx[i], 0);
        }

    }

    if (indexed) {

        // Store triangles without duplicates.

        for (int i = 0; i < bvh.getScene()->getNumTriangles(); ++i)
        {
            woopifyTri_bypass(bvh, i);
            if (m_woop[0].x == 0.0f)
                m_woop[0].x = 0.0f;

            triWoopData_v0.add(m_woop[0]);
            triWoopData_v1.add(m_woop[1]);
            triWoopData_v2.add(m_woop[2]);
        }
    }

    // Dummies.
    for (int i = 0; i < 7; ++i) {
        triWoopData_v0.add(0x80000000);
        triWoopData_v1.add(0x80000000);
        triWoopData_v2.add(0x80000000);
        triIndexData.add(0);

        nodeDataXY.add(Vec4i(0, 0, 0, 0));
        nodeDataZI.add(Vec4i(0, 0, 0, 0));
    }


    // Write to buffers.

    m_nodes_XY.resizeDiscard(nodeDataXY.getNumBytes());
    m_nodes_XY.set(nodeDataXY.getPtr(), nodeDataXY.getNumBytes());

    m_nodes_ZI.resizeDiscard(nodeDataZI.getNumBytes());
    m_nodes_ZI.set(nodeDataZI.getPtr(), nodeDataZI.getNumBytes());

    m_triWoop_v0.resizeDiscard(triWoopData_v0.getNumBytes());
    m_triWoop_v0.set(triWoopData_v0.getPtr(), triWoopData_v0.getNumBytes());

    m_triWoop_v1.resizeDiscard(triWoopData_v1.getNumBytes());
    m_triWoop_v1.set(triWoopData_v1.getPtr(), triWoopData_v1.getNumBytes());

    m_triWoop_v2.resizeDiscard(triWoopData_v2.getNumBytes());
    m_triWoop_v2.set(triWoopData_v2.getPtr(), triWoopData_v2.getNumBytes());

    m_nodes.discard();
    m_triWoop.discard();

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());

}


//------------------------------------------------------------------------

void CudaBVH::woopifyTri(const BVH& bvh, int idx)
{
    const Vec3i* triVtxIndex = (const Vec3i*)bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* vtxPos = (const Vec3f*)bvh.getScene()->getVtxPosBuffer().getPtr();
    const Vec3i& inds = triVtxIndex[bvh.getTriIndices()[idx]];
    const Vec3f& v0 = vtxPos[inds.x];
    const Vec3f& v1 = vtxPos[inds.y];
    const Vec3f& v2 = vtxPos[inds.z];

    Mat4f mtx;
    mtx.setCol(0, Vec4f(v0 - v2, 0.0f));
    mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
    mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
    mtx.setCol(3, Vec4f(v2, 1.0f));
    mtx = invert(mtx);

    m_woop[0] = Vec4f(mtx(2,0), mtx(2,1), mtx(2,2), -mtx(2,3));
    m_woop[1] = mtx.getRow(0);
    m_woop[2] = mtx.getRow(1);
}

void CudaBVH::woopifyTri_bypass(const BVH& bvh, int idx)
{
    const Vec3i* triVtxIndex = (const Vec3i*)bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* vtxPos = (const Vec3f*)bvh.getScene()->getVtxPosBuffer().getPtr();
    const Vec3i& inds = triVtxIndex[idx];
    const Vec3f& v0 = vtxPos[inds.x];
    const Vec3f& v1 = vtxPos[inds.y];
    const Vec3f& v2 = vtxPos[inds.z];

    Mat4f mtx;
    mtx.setCol(0, Vec4f(v0 - v2, 0.0f));
    mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
    mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
    mtx.setCol(3, Vec4f(v2, 1.0f));
    mtx = invert(mtx);

    m_woop[0] = Vec4f(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
    m_woop[1] = mtx.getRow(0);
    m_woop[2] = mtx.getRow(1);
}

//------------------------------------------------------------------------
