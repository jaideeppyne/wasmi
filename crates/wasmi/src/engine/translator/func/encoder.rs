use super::{Reset, ReusableAllocations};
use crate::{
    Engine,
    Error,
    engine::{
        TranslationError,
        executor::op_code_to_handler,
        translator::{
            comparator::UpdateBranchOffset,
            func::{
                LabelRef,
                LabelRegistry,
                labels::{Label, ResolvedLabelUser},
            },
        },
    },
    ir::{self, BlockFuel, BranchOffset, Decode as _, Encode as _, Op, OpCode},
};
use alloc::vec::Vec;
use core::{cmp, fmt, iter, marker::PhantomData};

/// Fuel amount required by certain operators.
type FuelUsed = u64;

/// A byte position within the encoded byte buffer.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BytePos(usize);

impl From<usize> for BytePos {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl From<BytePos> for usize {
    fn from(pos: BytePos) -> Self {
        pos.0
    }
}

/// A position within the encoded byte buffer and its known encoded type.
pub struct Pos<T> {
    /// The underlying byte position.
    value: BytePos,
    /// The type marker denoting what value type has been encoded.
    marker: PhantomData<fn() -> T>,
}

impl<T> From<BytePos> for Pos<T> {
    fn from(value: BytePos) -> Self {
        Self {
            value,
            marker: PhantomData,
        }
    }
}
impl<T> From<Pos<T>> for BytePos {
    fn from(pos: Pos<T>) -> Self {
        pos.value
    }
}
impl<T> Copy for Pos<T> {}
impl<T> Clone for Pos<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> PartialEq for Pos<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl<T> Eq for Pos<T> {}
impl<T> PartialOrd for Pos<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for Pos<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.value.cmp(&other.value)
    }
}
impl<T> fmt::Debug for Pos<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pos")
            .field("value", &self.value)
            .field("marker", &self.marker)
            .finish()
    }
}

#[derive(Debug, Default)]
pub struct EncodedOps {
    buffer: Vec<u8>,
    temp: Option<ReportingPos>,
    /// The positions of all encoded [`BranchOffset`]s and their branch operators.
    ///
    /// # Note
    ///
    /// This is required to relocate all encoded [`BranchOffset`]s into absolute
    /// branch targets once the encoded operators reside in their final, address
    /// stable allocation. Read more about this in [`EncodedOps::relocate_branch_offsets`].
    branch_offsets: Vec<EncodedBranchOffset>,
}

/// The position of an encoded [`BranchOffset`] and of its branch operator.
#[derive(Debug, Copy, Clone)]
struct EncodedBranchOffset {
    /// The position of the encoded item that stores the [`BranchOffset`].
    ///
    /// This is the position that the [`BranchOffset`] is relative to.
    src: BytePos,
    /// The position of the encoded [`BranchOffset`] itself.
    pos: Pos<BranchOffset>,
}

/// A [`Pos`] of an encoded item that needs to be reported back.
#[derive(Debug)]
enum ReportingPos {
    /// The temporary object is a [`BranchOffset`].
    BranchOffset(Pos<BranchOffset>),
    /// The temporary object is a [`BlockFuel`].
    BlockFuel(Pos<BlockFuel>),
}

impl Reset for EncodedOps {
    fn reset(&mut self) {
        self.buffer.clear();
        self.branch_offsets.clear();
    }
}

impl EncodedOps {
    /// Returns the next [`BytePos`].
    #[must_use]
    fn next_pos(&self) -> BytePos {
        BytePos::from(self.buffer.len())
    }

    /// Takes the reporting [`Pos`] if any exists.
    #[must_use]
    fn take_reporting_pos(&mut self) -> Option<ReportingPos> {
        self.temp.take()
    }

    /// Registers the encoded [`BranchOffset`] at `pos` of the branch operator at `src`.
    fn register_branch_offset(&mut self, src: BytePos, pos: Pos<BranchOffset>) {
        self.branch_offsets.push(EncodedBranchOffset { src, pos });
    }

    /// Truncates the buffer to `pos`.
    ///
    /// This clears everything that has been encoded to `self` after `pos`.
    fn truncate(&mut self, pos: impl Into<BytePos>) {
        let new_len = pos.into().0;
        debug_assert!(new_len <= self.buffer.len());
        self.buffer.truncate(new_len);
        // Note: branch operators are never staged and thus never truncated.
        //       This merely guards against future changes to this invariant.
        while let Some(last) = self.branch_offsets.last() {
            if BytePos::from(last.pos).0 < new_len {
                break;
            }
            self.branch_offsets.pop();
        }
    }
}

impl ir::Encoder for EncodedOps {
    type Pos = BytePos;
    type Error = TranslationError;

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<Self::Pos, Self::Error> {
        let pos = self.buffer.len();
        if self.buffer.try_reserve(bytes.len()).is_err() {
            return Err(TranslationError::OutOfSystemMemory);
        }
        self.buffer.extend(bytes);
        Ok(BytePos::from(pos))
    }

    fn encode_op_code(&mut self, code: OpCode) -> Result<Self::Pos, Self::Error> {
        encode_op_code(self, code)
    }

    fn branch_offset(
        &mut self,
        pos: Self::Pos,
        _branch_offset: BranchOffset,
    ) -> Result<(), Self::Error> {
        debug_assert!(self.temp.is_none());
        self.temp = Some(ReportingPos::BranchOffset(pos.into()));
        Ok(())
    }

    fn block_fuel(
        &mut self,
        pos: Self::Pos,
        _block_fuel: ir::BlockFuel,
    ) -> Result<(), Self::Error> {
        debug_assert!(self.temp.is_none());
        self.temp = Some(ReportingPos::BlockFuel(pos.into()));
        Ok(())
    }
}

/// Creates and encodes the buffer of encoded [`Op`]s for a function.
#[derive(Debug, Default)]
pub struct OpEncoder {
    /// The currently staged [`Op`].
    ///
    /// # Note
    ///
    /// - This allows the last [`Op`] to be peeked, inspected and manipulated.
    /// - For example, this is useful to perform op-code fusion or adjusting the result slot.
    ///
    /// # Invariant
    ///
    /// If `staged` is `Some`, the staged operator has already been encoded as the last [`Op`] in `ops`.
    staged: Option<StagedOp>,
    /// This is `true` if fuel metering is enabled.
    consume_fuel: bool,
    /// The list of constructed instructions and their parameters.
    ops: EncodedOps,
    /// Labels and label users for control flow and encoded branch operators.
    labels: LabelRegistry,
}

/// The staged [`Op`].
#[derive(Debug, Copy, Clone)]
pub struct StagedOp {
    /// The staged [`Op`].
    op: Op,
    /// The position of the encoded staged [`Op`].
    pos: Pos<Op>,
}

impl StagedOp {
    /// Creates a new [`StagedOp`] from `op`.
    pub fn new(op: Op, pos: Pos<Op>) -> Self {
        Self { op, pos }
    }

    /// Updates the current [`Op`] of `self` with `op`.
    ///
    /// # Note
    ///
    /// There is no need to update `pos` since it stays the same
    /// given that the staged [`Op`] is always last.
    pub fn update(&mut self, op: Op) {
        self.op = op;
    }
}

impl ReusableAllocations for OpEncoder {
    type Allocations = OpEncoderAllocations;

    fn into_allocations(self) -> Self::Allocations {
        Self::Allocations {
            ops: self.ops,
            labels: self.labels,
        }
    }
}

/// The reusable heap allocations of the [`OpEncoder`].
#[derive(Debug, Default)]
pub struct OpEncoderAllocations {
    /// The list of constructed instructions and their parameters.
    ops: EncodedOps,
    /// Labels and label users for control flow and encoded branch operators.
    labels: LabelRegistry,
}

impl Reset for OpEncoderAllocations {
    fn reset(&mut self) {
        self.ops.reset();
        self.labels.reset();
    }
}

impl OpEncoder {
    /// Creates a new [`OpEncoder`].
    pub fn new(engine: &Engine, alloc: OpEncoderAllocations) -> Self {
        let consume_fuel = engine.config().get_consume_fuel();
        Self {
            staged: None,
            consume_fuel,
            ops: alloc.ops,
            labels: alloc.labels,
        }
    }

    /// Allocates a new unpinned [`Label`].
    pub fn new_label(&mut self) -> LabelRef {
        self.labels.new_label()
    }

    /// Pins the [`Label`] at `lref` to the current encoded bytestream position.
    ///
    /// # Panics
    ///
    /// If there is a staged [`Op`].
    pub fn pin_label(&mut self, lref: LabelRef) -> Result<(), Error> {
        self.commit_staged_if_any()?;
        self.pad_to_op_alignment()?;
        let next_pos = Pos::from(self.ops.next_pos());
        self.labels.pin_label(lref, next_pos);
        Ok(())
    }

    /// Pins the [`Label`] at `lref` to the current encoded bytestream position if unpinned.
    ///
    /// # Note
    ///
    /// Does nothing if the label is already pinned.
    ///
    /// # Panics
    ///
    /// If there is a staged [`Op`].
    pub fn pin_label_if_unpinned(&mut self, lref: LabelRef) -> Result<(), Error> {
        self.commit_staged_if_any()?;
        self.pad_to_op_alignment()?;
        let next_pos = Pos::from(self.ops.next_pos());
        self.labels.pin_label_if_unpinned(lref, next_pos);
        Ok(())
    }

    /// Resolves the [`BranchOffset`] to `lref` from the current encoded bytestream position if `lref` is pinned.
    ///
    ///
    /// # Note
    ///
    /// Returns an uninitialized [`BranchOffset`] if `lref` refers to an unpinned [`Label`].
    ///
    /// # Panics
    ///
    /// If there is a staged [`Op`].
    fn try_resolve_label(&mut self, lref: LabelRef) -> BranchOffset {
        assert!(self.staged.is_none());
        let src = self.ops.next_pos();
        match self.labels.get_label(lref) {
            Label::Pinned(dst) => trace_branch_offset(src, dst),
            Label::Unpinned => BranchOffset::uninit(),
        }
    }

    /// Returns the staged [`Op`] if any.
    pub fn peek_staged(&self) -> Option<Op> {
        self.staged.map(|staged| staged.op)
    }

    /// Sets the staged [`Op`] to `new_staged` and encodes the previously staged [`Op`] if any.
    ///
    /// Returns the [`Pos<Op>`] of the staged [`Op`] if it was encoded.
    pub fn stage_op(&mut self, new_staged: Op) -> Result<(), Error> {
        self.commit_staged_if_any()?;
        self.pad_to_op_alignment()?;
        let pos = self.encode_impl(new_staged)?;
        self.staged = Some(StagedOp::new(new_staged, pos));
        Ok(())
    }

    /// Commits the staged [`Op`] if there is any.
    ///
    /// # Note
    ///
    /// - After this operation there will be no more staged [`Op`].
    /// - Does nothing if there is no staged [`Op`].
    ///
    /// # Panics (Debug)
    ///
    /// If the staged operator unexpectedly issued [`BranchOffset`] or [`BlockFuel`] fields.
    /// Those operators may never be staged and must be taken care of directly.
    pub fn commit_staged_if_any(&mut self) -> Result<(), Error> {
        self.staged = None;
        debug_assert!(self.ops.temp.is_none());
        Ok(())
    }

    /// Drops the staged [`Op`] without encoding it.
    ///
    /// # Panics
    ///
    /// If there was no staged [`Op`].
    pub fn drop_staged(&mut self) {
        let Some(staged) = self.staged.take() else {
            panic!("could not drop staged `Op` since there was none")
        };
        debug_assert!(self.staged.is_none());
        self.ops.truncate(staged.pos);
    }

    /// Replaces the staged [`Op`] with `new_staged`.
    ///
    /// - This does __not__ encode the currently staged [`Op`] but merely replaces it.
    /// - Returns the [`Pos<Op>`] of the newly staged [`Op`].
    ///
    /// # Panics (Debug)
    ///
    /// If there currently is no staged [`Op`] that can be replaced.
    pub fn replace_staged(&mut self, new_staged: Op) -> Result<(), Error> {
        let Some(staged) = self.staged.as_mut() else {
            panic!("expected a staged `Op` but found `None`")
        };
        staged.update(new_staged);
        self.ops.truncate(staged.pos);
        self.encode_impl(new_staged)?;
        Ok(())
    }

    /// Encodes an item of type `T` to the [`OpEncoder`] and returns its [`Pos`].
    pub fn encode_op(&mut self, op: Op) -> Result<Pos<Op>, Error> {
        self.commit_staged_if_any()?;
        self.pad_to_op_alignment()?;
        let pos = self.encode_impl(op)?;
        debug_assert!(self.ops.take_reporting_pos().is_none());
        debug_assert!(self.staged.is_none());
        Ok(pos)
    }

    /// Encodes an [`Op::ConsumeFuel`] operator to `self`.
    ///
    /// # Note
    ///
    /// Every [`Op::ConsumeFuel`] charges at least 1 unit of fuel out of caution.
    pub fn encode_consume_fuel_op(&mut self) -> Result<Option<Pos<BlockFuel>>, Error> {
        if !self.consume_fuel {
            return Ok(None);
        }
        let consumed_fuel = BlockFuel::from(1);
        self.commit_staged_if_any()?;
        self.pad_to_op_alignment()?;
        Op::consume_fuel(consumed_fuel).encode(&mut self.ops)?;
        let Some(ReportingPos::BlockFuel(pos)) = self.ops.take_reporting_pos() else {
            unreachable!("expected encoded `BlockFuel` entry but found none")
        };
        debug_assert!(self.staged.is_none());
        Ok(Some(pos))
    }

    /// Encodes a type with [`BranchOffset`] to the [`OpEncoder`] and returns its [`Pos<Op>`] and [`Pos<BranchOffset>`].
    pub fn encode_branch<T>(
        &mut self,
        dst: LabelRef,
        make_branch: impl FnOnce(BranchOffset) -> T,
    ) -> Result<(Pos<T>, Pos<BranchOffset>), Error>
    where
        T: ir::Encode + UpdateBranchOffset,
    {
        self.commit_staged_if_any()?;
        let offset = self.try_resolve_label(dst)?;
        let item = make_branch(offset);
        let pos_item = self.encode_impl(item)?;
        let Some(ReportingPos::BranchOffset(pos_offset)) = self.ops.take_reporting_pos() else {
            unreachable!("expected encoded position for `BranchOffset` entry but found none");
        };
        self.ops
            .register_branch_offset(BytePos::from(pos_item), pos_offset);
        if !self.labels.is_pinned(dst) {
            self.labels
                .new_user(dst, BytePos::from(pos_item), pos_offset);
        }
        debug_assert!(self.staged.is_none());
        Ok((pos_item, pos_offset))
    }

    /// Encodes an [`Op`] to the [`OpEncoder`] and returns its [`Pos<Op>`].
    ///
    /// # Note
    ///
    /// - Encodes `last` [`Op`] prior to `op` if `last` is `Some`.
    /// - After this call `last` will yield `None`.
    fn encode_impl<T>(&mut self, op: T) -> Result<Pos<T>, Error>
    where
        T: ir::Encode,
    {
        let pos = self.ops.next_pos();
        op.encode(&mut self.ops)?;
        Ok(Pos::from(pos))
    }

    /// Pads the encoded operator buffer with zero bytes until its length is [`Op`] aligned.
    ///
    /// This is used to encode new [`Op`] at properly aligned offsets within the buffer.
    /// The alignment is equal to the alignment of function pointers, e.g. `fn()`.
    ///
    /// Does nothing if the buffer is already aligned.
    pub fn pad_to_op_alignment(&mut self) -> Result<(), Error> {
        const ALIGN: usize = core::mem::align_of::<fn()>();
        if cfg!(feature = "indirect-dispatch") {
            // Only pad to alignment when `indirect-dispatch` is disabled.
            return Ok(());
        }
        let len = self.ops.buffer.len();
        let aligned_len = len.next_multiple_of(ALIGN);
        let padding_len = aligned_len - len;
        self.ops.buffer.extend(iter::repeat_n(0_u8, padding_len));
        Ok(())
    }

    /// Bumps consumed fuel for [`Op::ConsumeFuel`] at `fuel_pos` by `delta`.
    ///
    /// Does nothing if fuel metering is disabled.
    ///
    /// # Errors
    ///
    /// If consumed fuel is out of bounds after this operation.
    pub fn bump_fuel_consumption_by(
        &mut self,
        fuel_pos: Option<Pos<BlockFuel>>,
        delta: FuelUsed,
    ) -> Result<(), Error> {
        debug_assert_eq!(fuel_pos.is_some(), self.consume_fuel);
        let fuel_pos = match fuel_pos {
            None => return Ok(()),
            Some(fuel_pos) => fuel_pos,
        };
        self.ops
            .update_encoded(fuel_pos, |mut fuel| -> Option<BlockFuel> {
                fuel.bump_by(delta).ok()?;
                Some(fuel)
            });
        Ok(())
    }

    /// Returns an iterator yielding all encoded [`Op`]s of the [`OpEncoder`] as bytes.
    pub fn encoded_ops(&self) -> &[u8] {
        debug_assert!(self.staged.is_none());
        &self.ops.buffer[..]
    }

    /// Updates the branch offsets of all branch instructions inplace.
    ///
    /// # Panics
    ///
    /// If this is used before all branching labels have been pinned.
    pub fn update_branch_offsets(&mut self) {
        for user in self.labels.resolved_users() {
            let ResolvedLabelUser { src, dst, pos } = user;
            let offset = trace_branch_offset(src, dst);
            self.ops.update_branch_offset(pos, offset);
        }
    }

    /// Relocates all encoded [`BranchOffset`]s in `ops` into absolute branch targets.
    ///
    /// # Note
    ///
    /// - `ops` must be the final, address stable allocation storing a copy of the
    ///   encoded operators of `self`. Read more about this in [`BranchOffset`].
    /// - Must be called after [`OpEncoder::update_branch_offsets`] so that all
    ///   forward branch offsets are known.
    ///
    /// # Panics
    ///
    /// If a registered [`BranchOffset`] position is out of bounds for `ops`.
    pub fn relocate_branch_offsets(&self, ops: &mut [u8]) {
        self.ops.relocate_branch_offsets(ops);
    }
}

/// Error indicating that in-place updating of encoded items failed.
struct UpdateEncodedError<T> {
    /// The underlying kind of error.
    kind: UpdateEncodedErrorKind,
    /// The type that is decoded, updated and re-encoded.
    marker: PhantomData<fn() -> T>,
}

impl<T> From<UpdateEncodedErrorKind> for UpdateEncodedError<T> {
    fn from(kind: UpdateEncodedErrorKind) -> Self {
        Self {
            kind,
            marker: PhantomData,
        }
    }
}
impl<T> Clone for UpdateEncodedError<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for UpdateEncodedError<T> {}
impl<T> fmt::Debug for UpdateEncodedError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpdateEncodedError")
            .field("kind", &self.kind)
            .field("marker", &self.marker)
            .finish()
    }
}
impl<T> fmt::Display for UpdateEncodedError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self.kind {
            UpdateEncodedErrorKind::BufferOutOfBounds => "buffer out of bounds",
            UpdateEncodedErrorKind::FailedToDecode => "failed to decode",
            UpdateEncodedErrorKind::FailedToEncode => "failed to encode",
            UpdateEncodedErrorKind::FailedToUpdateEncoded => "failed to update encoded",
        };
        let type_name = core::any::type_name::<T>();
        write!(f, "{message}: {type_name}")
    }
}

/// Kinds of errors indicating that in-place updating of encoded items failed.
#[derive(Debug, Copy, Clone)]
enum UpdateEncodedErrorKind {
    /// Buffer is out of bounds for the position of the update.
    BufferOutOfBounds,
    /// Failed to decode the encoded item.
    FailedToDecode,
    /// Failed to encode the updated item.
    FailedToEncode,
    /// Failed to update the encoded item.
    FailedToUpdateEncoded,
}

impl EncodedOps {
    /// Updates the encoded [`BranchOffset`] at `pos` to `offset`.
    ///
    /// # Panics
    ///
    /// - If `pos` was out of bounds for `self`.
    /// - If the [`BranchOffset`] at `pos` failed to be decoded, updated or re-encoded.
    pub fn update_branch_offset(&mut self, pos: Pos<BranchOffset>, offset: BranchOffset) {
        self.update_encoded(pos, |_| Some(offset));
    }

    /// Relocates all encoded [`BranchOffset`]s in `ops` into absolute branch targets.
    ///
    /// # Note
    ///
    /// The `ops` buffer must be a copy of `self`'s encoded operators that resides
    /// in its final, address stable allocation. Since all encoded [`BranchOffset`]s
    /// are relative to the encoded item that stores them, their absolute branch
    /// target simply is `ops.as_ptr() + src + offset`.
    ///
    /// # Panics
    ///
    /// If a registered [`BranchOffset`] position is out of bounds for `ops`.
    fn relocate_branch_offsets(&self, ops: &mut [u8]) {
        debug_assert_eq!(self.buffer.len(), ops.len());
        // Note: the pointer's provenance is exposed here since the executor
        //       restores it via `ptr::with_exposed_provenance` upon branching.
        //       This mirrors how op-code handler pointers are encoded when the
        //       `indirect-dispatch` crate feature is disabled.
        let base = ops.as_mut_ptr().expose_provenance();
        for &EncodedBranchOffset { src, pos } in &self.branch_offsets {
            let at = usize::from(BytePos::from(pos));
            let Some(buffer) = ops.get_mut(at..) else {
                panic!("branch offset position is out of bounds: {at}")
            };
            let Ok(offset) = BranchOffset::decode(&mut &buffer[..]) else {
                panic!("failed to decode `BranchOffset` at: {at}")
            };
            // Note: an offset of 0 is a valid backwards branch, e.g. for `(loop (br 0))`,
            //       thus `BranchOffset::is_init` must not be asserted here.
            let target = base
                .wrapping_add(usize::from(src))
                .wrapping_add_signed(isize::from(offset));
            let target = BranchOffset::from(target as isize);
            if target.encode(&mut SliceEncoder::from(buffer)).is_err() {
                panic!("failed to encode branch target at: {at}")
            }
        }
    }

    /// Updates an encoded value `v` of type `T` at `pos` in-place using the result of `f(v)`.
    ///
    /// # Panics
    ///
    /// - If the underlying bytes buffer is out of bounds for `pos`.
    /// - If decodiing of `T` at `pos` fails.
    /// - If encodiing of `T` at `pos` fails.
    fn update_encoded<T>(&mut self, pos: Pos<T>, f: impl FnOnce(T) -> Option<T>)
    where
        T: ir::Encode + ir::Decode,
    {
        if let Err(error) = self
            .update_encoded_or_err(pos, f)
            .map_err(<UpdateEncodedError<T>>::from)
        {
            panic!("`OpEncoder::update_encoded` unexpectedly failed: {error}")
        }
    }

    /// Updates a value of type `T` at `pos` using `f` in the encoded buffer.
    ///
    /// # Errors
    ///
    /// - If the underlying bytes buffer is out of bounds for `pos`.
    /// - If decodiing of `T` at `pos` fails.
    /// - If encodiing of `T` at `pos` fails.
    /// - If `f(value)` returns `None` and thus updating failed.
    fn update_encoded_or_err<T>(
        &mut self,
        pos: Pos<T>,
        f: impl FnOnce(T) -> Option<T>,
    ) -> Result<(), UpdateEncodedErrorKind>
    where
        T: ir::Decode + ir::Encode,
    {
        let at = usize::from(BytePos::from(pos));
        let Some(buffer) = self.buffer.get_mut(at..) else {
            return Err(UpdateEncodedErrorKind::BufferOutOfBounds);
        };
        let Ok(decoded) = T::decode(&mut &buffer[..]) else {
            return Err(UpdateEncodedErrorKind::FailedToDecode);
        };
        let Some(updated) = f(decoded) else {
            return Err(UpdateEncodedErrorKind::FailedToUpdateEncoded);
        };
        if updated.encode(&mut SliceEncoder::from(buffer)).is_err() {
            return Err(UpdateEncodedErrorKind::FailedToEncode);
        }
        Ok(())
    }
}

/// Utility type to encode items to a slice of bytes.
pub struct SliceEncoder<'a> {
    /// The underlying bytes that will store the encoded items.
    bytes: &'a mut [u8],
}

/// An error that may occur upon encoding items to a byte slice.
#[derive(Debug, Copy, Clone)]
pub struct SliceEncoderError;

impl<'a> From<&'a mut [u8]> for SliceEncoder<'a> {
    fn from(bytes: &'a mut [u8]) -> Self {
        Self { bytes }
    }
}

impl<'a> ir::Encoder for SliceEncoder<'a> {
    type Pos = ();
    type Error = SliceEncoderError;

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<Self::Pos, Self::Error> {
        let Some(buffer) = self.bytes.get_mut(..bytes.len()) else {
            return Err(SliceEncoderError);
        };
        buffer.copy_from_slice(bytes);
        Ok(())
    }

    fn encode_op_code(&mut self, code: OpCode) -> Result<Self::Pos, Self::Error> {
        encode_op_code(self, code)
    }

    fn branch_offset(
        &mut self,
        _pos: Self::Pos,
        _branch_offset: BranchOffset,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn block_fuel(&mut self, _pos: Self::Pos, _block_fuel: BlockFuel) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Encodes an [`OpCode`] to a generic [`ir::Encoder`].
fn encode_op_code<E: ir::Encoder>(encoder: &mut E, code: OpCode) -> Result<E::Pos, E::Error> {
    match cfg!(feature = "indirect-dispatch") {
        true => {
            // Note: encoding for indirect-threading
            //
            // The op-codes are not resolved during translation time and must
            // be resolved during execution time. This decreases memory footprint
            // of the encoded IR at the cost of execution performance.
            u16::from(code).encode(encoder)
        }
        false => {
            // Note: encoding for direct-threading
            //
            // The op-codes are resolved during translation time (now) to their
            // underlying function pointers. This increases memory footprint
            // of the encoded IR but improves execution performance.
            (op_code_to_handler(code) as usize).encode(encoder)
        }
    }
}

/// Creates an initialized [`BranchOffset`] from `src` to `dst`.
///
/// # Note
///
/// [`BranchOffset`] is as wide as a pointer and thus can encode any offset within
/// the encoded operators buffer. Therefore this operation cannot fail.
fn trace_branch_offset(src: BytePos, dst: Pos<Op>) -> BranchOffset {
    // Note: Rust limits allocations to `isize::MAX` bytes, thus both byte
    //       positions always fit into an `isize` and cannot overflow.
    let src = usize::from(src) as isize;
    let dst = usize::from(BytePos::from(dst)) as isize;
    BranchOffset::from(dst.wrapping_sub(src))
}
